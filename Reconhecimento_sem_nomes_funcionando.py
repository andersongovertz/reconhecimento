import cv2
import numpy as np
import pandas as pd

# Define a função para encontrar o objeto circular na imagem


def find_circles(img, minDist, param1, param2, minRadius, maxRadius):
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1,
                               param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    if circles is not None:
        return np.uint16(np.around(circles))[0]
    else:
        return None


# Define os parâmetros para a detecção de círculos
minDist = 50
param1 = 50
param2 = 30
minRadius = 20
maxRadius = 25

# Inicializa o dataframe
df = pd.DataFrame(columns=['Frame', 'Objeto', 'Coordenadas'])

# Define o vídeo a ser lido
cap = cv2.VideoCapture('IMG_1706.mov')

circles_names = ['P7', 'P6', 'P5', 'P4', 'P3', 'P2', 'P1']


# Cria a janela para exibição do vídeo
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

# Loop para processar cada frame do vídeo
while True:
    ret, frame = cap.read()

    # Verifica se o frame foi lido corretamente
    if ret:

        # Converte o frame para tons de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Aplica um filtro de suavização Gaussiano
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Encontra os objetos circulares na imagem
        circles = find_circles(blurred, minDist, param1,
                               param2, minRadius, maxRadius)

        # Verifica se foram encontrados objetos
        if circles is not None:

            # Desenha um círculo em torno de cada objeto encontrado
            for i, circle in enumerate(circles):
                cv2.circle(frame, (circle[0], circle[1]),
                           circle[2], (0, 255, 0), 2)

                # Adiciona as informações do objeto ao dataframe
                df = df._append({'Frame': cap.get(cv2.CAP_PROP_POS_FRAMES),
                                 'Objeto': f'P{i+1}', 'Coordenadas': circle[:2]}, ignore_index=True)

        # Escreve a taxa de quadros no canto inferior esquerdo do vídeo
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cv2.putText(
            frame, f'FPS: {fps}', (50, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Exibe o frame na janela 'Video'
        cv2.imshow('Video', frame)

        # Verifica se a tecla 'Q' foi pressionada para sair do loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Sai do loop caso não haja mais frames para ler
    else:
        break

# Salva as informações do dataframe em um arquivo CSV
df.to_csv('informacoes_objetos_01.csv', index=False)

# Libera os recursos utilizados pelo OpenCV
cap.release()
cv2.destroyAllWindows()
