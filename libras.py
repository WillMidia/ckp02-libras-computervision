import cv2
import mediapipe as mp
import numpy as np
import math

class LibrasGestureRecognizer:
    def __init__(self):
        # Configuração do MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Mapeamento das letras para os métodos de detecção
        self.LETRAS_LIBRAS = {
            "A": self._detectar_letra_A,
            "B": self._detectar_letra_B,
            "C": self._detectar_letra_C,
            "D": self._detectar_letra_D,
            "E": self._detectar_letra_E,
            # Adicione os outros métodos conforme necessário
        }
    
    def _calcular_angulo(self, p1, p2, p3):
        """Calcula o ângulo entre três pontos."""
        vetor1 = np.array([p1.x - p2.x, p1.y - p2.y])
        vetor2 = np.array([p3.x - p2.x, p3.y - p2.y])
        return np.degrees(np.arctan2(np.cross(vetor1, vetor2), np.dot(vetor1, vetor2)))
    
    def _distancia_pontos(self, p1, p2):
        """Calcula a distância entre dois pontos."""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    def _esta_dobrado(self, landmark_base, landmark_ponta, threshold=0.1):
        """Verifica se o dedo está dobrado comparando a posição y (para imagem espelhada)."""
        return landmark_ponta.y > landmark_base.y - threshold
    
    def _esta_esticado(self, landmark_base, landmark_ponta, threshold=0.1):
        """Verifica se o dedo está esticado."""
        return landmark_ponta.y < landmark_base.y - threshold
    
    # Métodos de detecção para cada letra
    def _detectar_letra_A(self, landmarks):
        """Letra A: punho fechado - todos os dedos dobrados."""
        return all(
            self._esta_dobrado(landmarks[base], landmarks[tip])
            for base, tip in [
                (self.mp_hands.HandLandmark.INDEX_FINGER_MCP, self.mp_hands.HandLandmark.INDEX_FINGER_TIP),
                (self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP, self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP),
                (self.mp_hands.HandLandmark.RING_FINGER_MCP, self.mp_hands.HandLandmark.RING_FINGER_TIP),
                (self.mp_hands.HandLandmark.PINKY_MCP, self.mp_hands.HandLandmark.PINKY_TIP)
            ]
        )
    
    def _detectar_letra_B(self, landmarks):
        """Letra B: todos os dedos esticados."""
        return all(
            self._esta_esticado(landmarks[base], landmarks[tip])
            for base, tip in [
                (self.mp_hands.HandLandmark.INDEX_FINGER_MCP, self.mp_hands.HandLandmark.INDEX_FINGER_TIP),
                (self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP, self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP),
                (self.mp_hands.HandLandmark.RING_FINGER_MCP, self.mp_hands.HandLandmark.RING_FINGER_TIP),
                (self.mp_hands.HandLandmark.PINKY_MCP, self.mp_hands.HandLandmark.PINKY_TIP)
            ]
        )
    
    def _detectar_letra_C(self, landmarks):
        """Letra C: mão curvada em formato de C."""
        distancia = self._distancia_pontos(
            landmarks[self.mp_hands.HandLandmark.THUMB_TIP],
            landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        )
        dedos_dobrados = all(
            self._esta_dobrado(landmarks[base], landmarks[tip])
            for base, tip in [
                (self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP, self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP),
                (self.mp_hands.HandLandmark.RING_FINGER_MCP, self.mp_hands.HandLandmark.RING_FINGER_TIP),
                (self.mp_hands.HandLandmark.PINKY_MCP, self.mp_hands.HandLandmark.PINKY_TIP)
            ]
        )
        return distancia < 0.15 and dedos_dobrados
    
    def _detectar_letra_D(self, landmarks):
        """Letra D: indicador esticado e os outros dedos dobrados."""
        indicador_esticado = self._esta_esticado(
            landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_MCP],
            landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        )
        outros_dobrados = all(
            self._esta_dobrado(landmarks[base], landmarks[tip])
            for base, tip in [
                (self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP, self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP),
                (self.mp_hands.HandLandmark.RING_FINGER_MCP, self.mp_hands.HandLandmark.RING_FINGER_TIP),
                (self.mp_hands.HandLandmark.PINKY_MCP, self.mp_hands.HandLandmark.PINKY_TIP)
            ]
        )
        return indicador_esticado and outros_dobrados
    
    def _detectar_letra_E(self, landmarks):
        """Letra E: similar à letra A (punho fechado)."""
        return self._detectar_letra_A(landmarks)
    
    def reconhecer_letra(self, landmarks):
        """Reconhece a letra baseada nos landmarks da mão."""
        for letra, detector in self.LETRAS_LIBRAS.items():
            if detector(landmarks):
                return letra
        return None
    
    def run(self):
        """Executa a aplicação de reconhecimento de Libras."""
        cap = cv2.VideoCapture(0)
        nome_escrito = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Inverter o frame
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detectar mãos
            resultados = self.hands.process(rgb_frame)
            
            # Interface
            cv2.putText(frame, "Escreva seu nome em Libras!", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Nome: {''.join(nome_escrito)}", (20, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "[ESPACO] Confirmar | [R] Resetar | [Q] Sair", 
                        (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            letra_atual = None
            if resultados.multi_hand_landmarks:
                for hand_landmarks in resultados.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    letra_atual = self.reconhecer_letra(hand_landmarks.landmark)
                    
                    if letra_atual:
                        cv2.putText(frame, f"Letra detectada: {letra_atual}", 
                                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            key = cv2.waitKey(10)
            if key == ord(' ') and letra_atual:
                nome_escrito.append(letra_atual)
            elif key == ord('r'):
                nome_escrito.clear()
            elif key == ord('q'):
                break
            
            cv2.imshow('Reconhecimento de Libras', frame)
        
        cap.release()
        cv2.destroyAllWindows()

# Executar a aplicação
if __name__ == "__main__":
    recognizer = LibrasGestureRecognizer()
    recognizer.run()
