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

# Define o diâmetro dos círculos em centímetros
diametro_cm = 25

# Define o número de faixas verticais
num_faixas = 7

# Inicializa o dataframe
df = pd.DataFrame(columns=['Frame', 'Objeto', 'Coordenadas', 'Aceleracao'])

# Define o vídeo a ser lido
cap = cv2.VideoCapture('IMG_1706.mov')

# Define as nomenclaturas dos círculos
circles_names = ['P7', 'P6', 'P5', 'P4', 'P3', 'P2', 'P1']

# Calcula a largura de cada faixa vertical
largura_faixa = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / num_faixas)

# Cria a janela para exibição do vídeo
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

# Inicializa a variável de posição do último círculo detectado
ultima_posicao = [0] * num_faixas

# Loop para processar cada frame do vídeo
while True:
    ret, frame = cap.read()

    # Verifica se o frame foi lido corretamente
    if ret:
        # Converte o frame para tons de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Aplica um filtro de suavização Gaussiano
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Loop através das faixas verticais
        for i in range(num_faixas):
            # Define as coordenadas da faixa vertical atual
            x_inicio = i * largura_faixa
            x_fim = (i + 1) * largura_faixa

            # Recorta a faixa vertical da imagem
            faixa = blurred[:, x_inicio:x_fim]

            # Encontra os objetos circulares na faixa vertical
            circles = find_circles(
                faixa, minDist, param1, param2, minRadius, maxRadius)

            # Verifica se foram encontrados objetos na faixa
            if circles is not None:
                # Escala para converter pixels em centímetros
                escala = diametro_cm / circles[0][2]

                # Loop através de cada círculo detectado na faixa
                for circle in circles:
                    # Obtém as coordenadas do círculo no frame completo
                    x = circle[0] + x_inicio
                    y = circle[1]

                    # Calcula a aceleração vertical
                    aceleracao = (y - ultima_posicao[i]) * escala

                    # Atualiza a posição do último círculo detectado na faixa
                    ultima_posicao[i] = y

                    # Desenha um círculo em torno do objeto encontrado
                    cv2.circle(frame, (x, y), circle[2], (0, 255, 0), 2)

                    # Exibe a nomenclatura acima do círculo
                    cv2.putText(
                        frame, circles_names[i], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Adiciona as informações do objeto ao dataframe
                    df = df._append({'Frame': cap.get(cv2.CAP_PROP_POS_FRAMES),
                                    'Objeto': circles_names[i], 'Coordenadas': (x, y), 'Aceleracao': aceleracao},
                                    ignore_index=True)

        # Calcula o tempo de vídeo executado e restante
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        video_time = current_frame / fps
        remaining_time = (total_frames - current_frame) / fps

        # Exibe os frames por segundo no canto superior esquerdo do vídeo
        cv2.putText(frame, f'FPS: {fps}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Exibe o tempo de vídeo executado e restante no canto inferior esquerdo do vídeo
        cv2.putText(frame, f'Tempo Executado: {video_time:.2f} s', (50, frame.shape[0] - 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f'Tempo Restante: {remaining_time:.2f} s', (50, frame.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

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
