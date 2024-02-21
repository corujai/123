import cv2
import numpy as np
import time
import os
import TrajectoryManager as tm
from ultralytics import YOLO
#from ultralytics.utils.plotting import Annotator
from numpy import array
def distancia_entre_pontos(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


minimap_size = (180, 200, 3)
minimap = np.zeros(minimap_size, dtype=np.uint8)
contagem_de_pontos_A = 0
contagem_de_pontos_B = 0
actual_lista_pontos = None
tempo_espera_apos_gol = 60  # Tempo de espera em segundos após um gol
ultimo_tempo_pontuacao = time.time()  # Inicializa com o tempo atual
ultimo_tempo_gol = 0
auxiliar = []
auxiliar_bola = 0

print(os.getcwd())

#cap = cv.VideoCapture('rtsp://playtime.cctvddns.net:554/profile1')
cap = cv2.VideoCapture("recordfinal_9.mp4")
bgSubtractor = cv2.createBackgroundSubtractorKNN(history = 10, dist2Threshold = 200.0)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

kernel_size = 11
kernel_dilation = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))
kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

frame_count = 0
trajectory_image = np.zeros([360, 640, 3], np.uint8)
point_image = np.zeros([360, 640, 3], np.uint8)


manager = tm.TrajectoryManager()
cv2.namedWindow('raw', cv2.WINDOW_NORMAL)
cv2.resizeWindow('raw', 640, 360)

_, one = cap.read()
past = one
present = past
future = past

minima_distancia_entre_coordenadas = 150

while True:
    start = time.time()
    past = present
    present = future
    _, frame = cap.read()
    future = frame

    blur = cv2.GaussianBlur(frame, (7, 7), 0)

    # Background
    fgmask = bgSubtractor.apply(blur)
    blank_image = np.zeros(fgmask.shape, np.uint8)

    # Background
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel_dilation)

    point_image = cv2.addWeighted(point_image, 0.9, np.zeros(frame.shape, np.uint8), 0.1, 0)

    # print("frame_count :", frame_count)
    frame_count += 1

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask)

    # print(len(centroids))

    points = []
    for index, centroid in enumerate(centroids):
        if stats[index][0] == 0 and stats[index][1] == 0:
            continue
        if np.any(np.isnan(centroid)):
            continue

        x, y, width, height, area = stats[index]
        centerX, centerY = int(centroid[0]), int(centroid[1])

        area_ratio = area / (width * height)
        aspect_ratio = width / height
        # print(x, y, area, width * height, area_ratio)

        # if area > 2 and area < 2000:
        if area_ratio > 0.6 and aspect_ratio > 0.333 and aspect_ratio < 3.0 and area < 500 and fgmask[
            centerY, centerX] == 255:

            cv2.circle(frame, (centerX, centerY), 1, (30, 255, 255), 1)
            # cv2.rectangle(frame, (x-1, y-1), (x-1 + width+2, y-1 + height+2), (0, 0, 255))

            # TODO
            # cv2.rectangle(frame, (x - 1, y - 1), (x - 1 + width + 2, y - 1 + height + 2), (0, 255, 0))

            point_image[centerY, centerX] = (255, 255, 255)
            points.append(np.array([centerY, centerX]))

            for pixel_y in range(y, y + height):
                for pixel_x in range(x, x + width):

                    if fgmask[pixel_y, pixel_x] >= 0:
                        # frame[pixel_y, pixel_x] = [0, 255, 0]
                        blank_image[pixel_y, pixel_x] = 255

        # else :

        #   cv2.rectangle(frame, (x - 1, y - 1), (x - 1 + width + 2, y - 1 + height + 2), (0, 0, 255))

    manager.setPointsFrame(points)

    for trajectory in manager.getTrajectorys():

        points = trajectory.getPoints()

        # Adicionando texto ao frame
        # texto_velocidades = ', '.join([f'{velocidade:.2f} m/s' for velocidade in velocidades])

        if len(points) < 2:
            continue

        for index, point in enumerate(points):
            if point[0] < 360 and point[1] < 640:

                trajectory_image[point[0], point[1]] = (0, 255, 0)
                cv2.circle(frame, (point[1], point[0]), 2, (255, 255, 255), 2)

                if index >= 2:
                    cv2.line(frame, (points[index - 1][1], points[index - 1][0]), (point[1], point[0]), (255, 255, 0),
                             1)

    delta_plus = cv2.absdiff(present, past)
    delta_0 = cv2.absdiff(future, past)
    delta_minus = cv2.absdiff(present, future)

    gray_plus = cv2.cvtColor(delta_plus, cv2.COLOR_BGR2GRAY)
    gray_0 = cv2.cvtColor(delta_0, cv2.COLOR_BGR2GRAY)
    gray_minus = cv2.cvtColor(delta_minus, cv2.COLOR_BGR2GRAY)

    th = 100

    fplus = cv2.threshold(gray_plus, th, 255, cv2.THRESH_BINARY)[1]
    f0 = cv2.threshold(gray_0, th, 255, cv2.THRESH_BINARY)[1]
    fminus = cv2.threshold(gray_minus, th, 255, cv2.THRESH_BINARY)[1]

    finalp1 = cv2.bitwise_or(fplus, f0)
    finalp2 = cv2.bitwise_or(finalp1, fminus)

    # Encontre contornos na imagem binária final
    contours, _ = cv2.findContours(finalp2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterar sobre os contornos e desenhar os centros e coordenadas
    frame_contours = frame.copy()
    coordenadas_exibidas = []

    ret, frame = cap.read()
    resize_scale = 640. / float(frame.shape[1])

    frame = cv2.resize(frame, (640, 360))

    blur = cv2.GaussianBlur(frame, (7, 7), 0)

    # Background
    fgmask = bgSubtractor.apply(blur)
    blank_image = np.zeros(fgmask.shape, np.uint8)

    # Background
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel_dilation)


            # cv2.imshow('Trajetoria', trajectory_image)


    # Adicionando texto ao frame


    cv2.putText(frame, f'Equipa A: {contagem_de_pontos_A}', (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)
    cv2.putText(frame, f'Equipa B: {contagem_de_pontos_B}', (450, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)



    # Desenhando o maior contorno no frame

    x1, y1 = 130, 82
    x1_b, y1_b = 170, 57
    x1_c, y1_c = 380, 100
    comprimento = 265
    comprimento_b = 289
    comprimento_c = 510

    # Calcule as coordenadas do ponto final com base no ângulo de 110 graus
    angulo = 110
    angulo_b = 33
    angulo_c = 125
    x2 = int(x1 + comprimento * np.cos(np.radians(angulo)))
    y2 = int(y1 + comprimento * np.sin(np.radians(angulo)))

    x2_b = int(x1_b + comprimento_b * np.cos(np.radians(angulo_b)))
    y2_b = int(y1_b + comprimento_b * np.sin(np.radians(angulo_b)))

    x2_c = int(x1_c + comprimento_c * np.cos(np.radians(angulo_c)))
    y2_c = int(y1_c + comprimento_c * np.sin(np.radians(angulo_c)))

    # Desenhe a linha na imagem
    cor_linha = (30, 255, 255)  # Cor da linha em BGR (verde neste exemplo)
    espessura_linha = 2
    # cv2.line(frame, (y1, x1), (y2, x2), cor_linha, espessura_linha)
    # cv2.line(frame, (x1_b, y1_b), (x2_b, y2_b), cor_linha, espessura_linha)
    # cv2.line(img, (y1_c, x1_c), (y2_c, x2_c), cor_linha, espessura_linha)

    # A linha
    # cv2.line(frame, (0, 353), (frame.shape[1], 353), (255, 0, 0), 2)

    # results = model.predict(frame)
    #annotator = Annotator(frame)

    # Clear the minimap by creating a new black image
    minimap = np.zeros(minimap_size, dtype=np.uint8)

    """
    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]
            bb = box.xywh[0]
            x, y, w, h = map(int, bb)


            if b[1] > 5:
                raio = 5
                centro_circulo = (x, y + h // 2)

                # Update minimap by adding circles instead of heatmap
                # Convert coordinates to fit the minimap size
                x_minimap = int(centro_circulo[0] * minimap_size[0] / frame.shape[1])
                y_minimap = int(centro_circulo[1] * minimap_size[1] / frame.shape[0])

                # Draw a white circle on the minimap
                if b[1] < 50:

                    cv2.circle(minimap, (x_minimap, y_minimap), raio, (0, 0, 255), -1)
                else:

                    cv2.circle(minimap, (x_minimap, y_minimap), raio, (0, 255, 0), -1)

            if box.cls == 2:
                if b[1] < 155:
                    #c = box.cls
                    annotator.box_label(b, "Equipa A", color=(0, 0, 255))

                else:
                    #c = box.cls
                    annotator.box_label(b, "Equipa B", color=(0, 255, 0))
            elif box.cls == 1:
                annotator.box_label(b, "Bola", color=(255, 255, 0))





            #frame[:180, :200] = minimap



    #cv2.imshow('processed', fgmask)
    #cv2.imshow('point', point_image)
    """

    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])


            adicionar_coordenada = True
            for coord_x, coord_y in coordenadas_exibidas:

                if distancia_entre_pontos(cX, cY, coord_x, coord_y) < minima_distancia_entre_coordenadas:
                        adicionar_coordenada = False
                        break


            if adicionar_coordenada:
                coordenadas_exibidas.append((cX, cY))
                cv2.drawContours(frame_contours, [contour], -1, (0, 255, 0), 2)
                cv2.circle(frame_contours, (cX, cY), 7, (0, 255, 0), -1)
                cv2.putText(frame_contours, f'({cX}, {cY})', (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar imagem com contornos e coordenadas
    cv2.imshow('hope', frame_contours)

    print(time.time() - start)

    k = cv2.waitKey(10) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()
