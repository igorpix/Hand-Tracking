import cv2  # Importa OpenCV para capturar e processar imagens da câmera
import mediapipe as mp  # Importa MediaPipe para detecção e rastreamento de mãos

# Inicializa a captura de vídeo (0 para a câmera principal, pode testar com 1, 2 se necessário)
video = cv2.VideoCapture(0)

# Verifica se a câmera abriu corretamente
if not video.isOpened():
    print("Erro: Não foi possível acessar a câmera.")
    exit()

# Inicializa o MediaPipe Hands
hands = mp.solutions.hands  # Configuração do modelo de detecção de mãos
Hands = hands.Hands(max_num_hands=1)  # Permite detectar até 1 mão
mpDraw = mp.solutions.drawing_utils  # Ferramenta para desenhar os pontos e conexões da mão

while True:
    success, img = video.read()  # Captura um frame da câmera
    
    # Verifica se a imagem foi capturada corretamente
    if not success or img is None:
        print("Erro ao capturar frame da câmera!")
        continue  # Pula para a próxima iteração do loop

    # Converte a imagem de BGR (padrão OpenCV) para RGB (usado pelo MediaPipe)
    frameRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = Hands.process(frameRGB)  # Processa a imagem para detectar mãos
    handPoints = results.multi_hand_landmarks  # Lista de mãos detectadas
    handedness = results.multi_handedness  # Informação se é mão direita ou esquerda
    
    h, w, _ = img.shape  # Obtém altura (h), largura (w) da imagem
    pontos = []  # Lista para armazenar os pontos da mão

    if handPoints:  # Se alguma mão for detectada
        for i, points in enumerate(handPoints):  # Para cada mão detectada
            mpDraw.draw_landmarks(img, points, hands.HAND_CONNECTIONS)  # Desenha a mão

            # Identifica se a mão detectada é direita ou esquerda
            if handedness:
                label = handedness[i].classification[0].label  # 'Right' ou 'Left'
                is_right_hand = label == "Right"  # True se for mão direita

            # Coleta os pontos da mão e os armazena na lista pontos
            for id, cord in enumerate(points.landmark):
                cx, cy = int(cord.x * w), int(cord.y * h)  # Converte coordenadas normalizadas em pixels
                pontos.append((cx, cy))  # Adiciona ponto na lista

            # Índices dos dedos (indicador, médio, anelar, mindinho)
            dedos = [8, 12, 16, 20]
            contador = 0  # Variável para contar os dedos levantados

            if pontos:
                # Contagem do polegar (diferente para mão direita e esquerda)
                if is_right_hand:
                    if pontos[4][0] < pontos[3][0]:  # Para a mão direita, polegar à esquerda do dedo indicador
                        contador += 1
                else:
                    if pontos[4][0] > pontos[3][0]:  # Para a mão esquerda, polegar à direita do dedo indicador
                        contador += 1

                # Contagem dos outros dedos (se a ponta estiver acima da articulação)
                for x in dedos:
                    if pontos[x][1] < pontos[x - 2][1]:
                        contador += 1

            # Desenha um retângulo azul para exibir a contagem de dedos
            cv2.rectangle(img, (80, 10), (200, 110), (255, 0, 0), -1)
            cv2.putText(img, str(contador), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 5)

    # Exibe a imagem com a contagem de dedos
    cv2.imshow('Imagem', img)
    
    # Fecha o programa quando a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a câmera e fecha as janelas
video.release()
cv2.destroyAllWindows()
