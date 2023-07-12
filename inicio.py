import cv2
import mediapipe as mp
import pyautogui

# Carregar o modelo de reconhecimento de mãos
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2)

# Inicializar o vídeo
video_capture = cv2.VideoCapture(0)  # Usar 0 para a câmera padrão

# Configurar parâmetros para controle de volume
volume_factor = 100  # Fator de ajuste do volume

# Estado atual dos dedos
finger_state = {
    'thumb': False,
    'index': False,
    'middle': False,
    'ring': False,
    'pinky': False
}

# Posição inicial dos dedos
finger_position = {
    'thumb': 0,
    'index': 0,
    'middle': 0,
    'ring': 0,
    'pinky': 0
}

# Variável de estado para controle do gesto de pular músicas
skip_music = False

while True:
    # Capturar o frame do vídeo
    ret, frame = video_capture.read()
    if not ret:
        break

    # Converter o frame para RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar as mãos no frame
    results = mp_hands.process(frame_rgb)

    # Verificar se mãos foram detectadas
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extrair os pontos-chave das mãos
            for landmark in hand_landmarks.landmark:
                # Coordenadas normalizadas dos pontos-chave
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                # Desenhar um círculo nos pontos-chave das mãos
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                # Verificar o estado dos dedos
                if landmark == hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]:
                    if landmark.y < hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].y:
                        finger_state['thumb'] = True
                    else:
                        finger_state['thumb'] = False
                        finger_position['thumb'] = landmark.x
                elif landmark == hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]:
                    if landmark.y < hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].y:
                        finger_state['index'] = True
                    else:
                        finger_state['index'] = False
                        finger_position['index'] = landmark.x
                elif landmark == hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]:
                    if landmark.y < hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].y:
                        finger_state['middle'] = True
                    else:
                        finger_state['middle'] = False
                        finger_position['middle'] = landmark.x
                elif landmark == hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]:
                    if landmark.y < hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].y:
                        finger_state['ring'] = True
                    else:
                        finger_state['ring'] = False
                        finger_position['ring'] = landmark.x
                elif landmark == hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]:
                    if landmark.y < hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].y:
                        finger_state['pinky'] = True
                    else:
                        finger_state['pinky'] = False
                        finger_position['pinky'] = landmark.x

            # Verificar o gesto de aumentar o volume (indicador estendido e outros dedos fechados)
            if finger_state['index'] and not finger_state['thumb'] and not finger_state['middle'] and \
                    not finger_state['ring'] and not finger_state['pinky']:
                # Aumentar o volume simulando a tecla de aumento do volume
                pyautogui.press('volumeup')

            # Verificar o gesto de diminuir o volume (mão fechada e todos os dedos apontando para baixo)
            if not finger_state['thumb'] and not finger_state['index'] and not finger_state['middle'] and \
                    not finger_state['ring'] and not finger_state['pinky']:
                # Diminuir o volume simulando a tecla de diminuição do volume
                pyautogui.press('volumedown')

            # Verificar o gesto de pular para a próxima música (indicador apontado para a direita)
            if finger_position['index'] > finger_position['thumb'] and \
                    finger_position['index'] > finger_position['middle'] and \
                    finger_position['index'] > finger_position['ring'] and \
                    finger_position['index'] > finger_position['pinky']:
                if not skip_music:
                    # Pular para a próxima música simulando a tecla de avançar
                    pyautogui.press('nexttrack')
                    # Atualizar o estado para evitar pular várias músicas seguidas
                    skip_music = True
            else:
                # Atualizar o estado para permitir pular músicas novamente
                skip_music = False

    # Exibir o frame com os pontos-chave das mãos
    cv2.imshow('Reconhecimento de Movimentos das Mãos', frame)

    # Sair do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
video_capture.release()
cv2.destroyAllWindows()
