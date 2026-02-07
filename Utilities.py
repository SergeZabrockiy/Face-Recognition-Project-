import cv2
import numpy as np
import math

class Utilities:
    
    @classmethod
    def create_heatmap(self, size, landmark, sigma=2):
        """
        Создаёт один heatmap с гауссовым ядром вокруг точки.

        :param size: (height, width) — размер heatmap'а
        :param landmark: (x, y) — координаты точки
        :param sigma: стандартное отклонение гауссиана
        :return: heatmap массив [H, W]
        """
        x, y = landmark
        h, w = size

        # Обрезаем координаты, чтобы не выйти за пределы изображения
        x = min(max(0, int(x)), w - 1)
        y = min(max(0, int(y)), h - 1)

        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        heatmap = np.exp(-((yy - y)**2 + (xx - x)**2) / (2 * sigma**2))
        return heatmap

    @classmethod
    def landmarks_to_heatmaps(self, image_shape, landmarks, sigma=2):
        """
        Преобразует список из N точек в набор из N heatmap'ов.

        :param image_shape: исходный размер изображения (H, W)
        :param landmarks: список из N пар координат [(x1, y1), (x2, y2), ..., (xN, yN)]
        :param sigma: стандартное отклонение гауссиана
        :return: массив heatmap'ов вида [N, H, W]
        """
        heatmaps = []
        for (x, y) in landmarks:
            hm = self.create_heatmap(image_shape, (x, y), sigma=sigma)
            heatmaps.append(hm)
        return np.array(heatmaps)
    
    @classmethod
    def find_peaks(self, heatmap, threshold):
        """
        Находит локальные максимумы (пики) на тепловой карте (heatmap).
        
        Params:
            heatmap: 2D-массив (H x W) — тепловая карта для одного класса/ключевой точки
            threshold: float — минимальный порог уверенности для рассмотрения пика
        
        
        Returns:
            Список словарей [{'coord': (x, y), 'value': val}], отсортированный по убыванию уверенности
        """
        h, w = heatmap.shape
        peaks = []
        # Проходим по всем пикселям, кроме границ (чтобы избежать выхода за пределы при проверке соседей)
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                val = heatmap[y, x]
                if val > threshold:
                    if (val >= heatmap[y-1, x] and val >= heatmap[y+1, x] and
                        val >= heatmap[y, x-1] and val >= heatmap[y, x+1] and
                        val >= heatmap[y-1, x-1] and val >= heatmap[y-1, x+1] and
                        val >= heatmap[y+1, x-1] and val >= heatmap[y+1, x+1]):
                        peaks.append({'coord': (x, y), 'value': val})
            # Сортируем пики по убыванию уверенности (значения heatmap)
            return sorted(peaks, key=lambda p: p['value'], reverse=True)

    @classmethod
    def post_process_landmarks(self, heatmaps_batch, scale_factor, confidence_threshold=0.1):
        """
        Пост‑обработка batch тепловых карт для получения согласованных координат ключевых точек лица.
        
        Params:
            heatmaps_batch: тензор формы (B, C, H, W) — batch тепловых карт (B: размер батча, 
                            C: число классов/ключевых точек, H/W: высота/ширина карт)
            scale_factor: float — коэффициент масштабирования координат (учитывает пулинг в сети)
            confidence_threshold: float — минимальный порог уверенности для пика (по умолчанию 0.1)
        
        
        Returns:
            final_landmarks_batch: numpy-массив формы (B, C, 2) — финальные координаты ключевых точек
                                для каждого изображения в батче
        """
        # Получаем размеры батча, числа классов и разрешения тепловых карт
        batch_size, num_classes, h, w = heatmaps_batch.shape
        # Инициализируем массив для финальных координат (B, C, 2): 2 — x и y
        final_landmarks_batch = np.zeros((batch_size, num_classes, 2))
        # Задаём индексы для ключевых точек (предполагаемый порядок в heatmaps)
        left_eye, right_eye, nose, left_month, right_mouth = 0, 1, 2, 3, 4

        # Обрабатываем каждое изображение в батче
        for i in range(batch_size):
            # Переводим тепловые карты текущего изображения в numpy
            heatmaps = heatmaps_batch[i].cpu().numpy()
            # Список списков: для каждой ключевой точки — список кандидатов (пиков)
            all_candidates = [[] for _ in range(num_classes)]
            for j in range(num_classes):
                candidates = self.find_peaks(heatmaps[j], confidence_threshold)
                # Если пиков не найдено — берём максимум heatmap как запасной вариант                
                if not candidates:
                    # Находим координаты максимума на карте
                    y_max, x_max = np.unravel_index(np.argmax(heatmaps[j]), (h, w))
                    # Добавляем максимум как единственный кандидат
                    candidates.append({'coord': (x_max, y_max), 'value': heatmaps[j, y_max, x_max]})
                all_candidates[j] = candidates
            # Выбираем лучший нос — самый уверенный пик (первый в списке кандидатов)
            best_nose = all_candidates[nose][0]['coord']
            
            # Ищем лучшую пару глаз (левый < правый по x, оба выше носа)
            best_eye_pair = {'left': None, 'right': None, 'score': -1}
            for le_cand in all_candidates[left_eye]:
                for re_cand in all_candidates[right_eye]:
                    if le_cand['coord'][0] < re_cand['coord'][0]:
                        if le_cand['coord'][1] < best_nose[1] and re_cand['coord'][1] < best_nose[1]:
                            score = le_cand['value'] + re_cand['value']
                            if score > best_eye_pair['score']:
                                best_eye_pair['score'] = score
                                best_eye_pair['left'] = le_cand['coord']
                                best_eye_pair['right'] = re_cand['coord']
            if best_eye_pair['left'] is None:
                best_eye_pair['left'] = all_candidates[left_eye][0]['coord']
                best_eye_pair['right'] = all_candidates[right_eye][0]['coord']
                if best_eye_pair['left'][0] > best_eye_pair['right'][0]:
                    best_eye_pair['left'],  best_eye_pair['right'] =  best_eye_pair['right'],  best_eye_pair['left']

            # Ищем лучшую пару уголков рта (левый < правый по x, оба ниже носа)        
            best_mouth_pair = {'left': None, 'right': None, 'score': -1}
            for lm_cand in all_candidates[left_month]:
                for rm_cand in all_candidates[right_mouth]:
                    if lm_cand['coord'][0] < rm_cand['coord'][0]:
                        if lm_cand['coord'][1] > best_nose[1] and rm_cand['coord'][1] > best_nose[1]:
                            score = lm_cand['value'] + rm_cand['value']
                            if score > best_mouth_pair['score']:
                                best_mouth_pair['score'] = score
                                best_mouth_pair['left'] = lm_cand['coord']
                                best_mouth_pair['right'] = rm_cand['coord']
            if best_mouth_pair['left'] is None:
                best_mouth_pair['left'] = all_candidates[left_month][0]['coord']
                best_mouth_pair['right'] = all_candidates[right_mouth][0]['coord']
                if best_mouth_pair['left'][0] > best_mouth_pair['right'][0]:
                    best_mouth_pair['left'],  best_mouth_pair['right'] =  best_mouth_pair['right'],  best_mouth_pair['left']

            # Собираем финальные координаты ключевых точек
            final_landmarks = np.zeros((num_classes, 2))
            final_landmarks[left_eye] = best_eye_pair['left']
            final_landmarks[right_eye] = best_eye_pair['right']
            final_landmarks[nose] = best_nose
            final_landmarks[left_month] = best_mouth_pair['left']
            final_landmarks[right_mouth] = best_mouth_pair['right']

            # Масштабируем координаты (учитываем пулинг в сети)
            final_landmarks_batch[i] = final_landmarks * scale_factor

        return final_landmarks_batch
    
    @classmethod
    def align_face_by_rotation(self, image, landmarks, desired_face_width=256):
        """
        Выравнивает лицо по горизонтали глаз с помощью аффинного преобразования.
        
        Params:
            image: numpy-массив (H, W, 3) — исходное изображение лица
            landmarks: список/массив (5, 2) — координаты ключевых точек:
                    [0] — левый глаз, [1] — правый глаз, [2] — нос, [3][4] — уголки рта
            desired_face_width: int — желаемая ширина выходного изображения (квадрат)
        
        
        Returns:
            aligned_face: numpy-массив (desired_face_width, desired_face_width, 3) — выровненное лицо
            dist_zero: bool — флаг, указывающий на нулевую дистанцию между глазами (ошибка)
        """
        dist_zero = False  # Флаг ошибки: если расстояние между глазами = 0

        left_eye = landmarks[0]   # Координаты левого глаза (x, y)
        right_eye = landmarks[1] # Координаты правого глаза (x, y)


        # Вычисляем вектор между глазами
        dY = right_eye[1] - left_eye[1]  # Разница по вертикали
        dX = right_eye[0] - left_eye[0]  # Разница по горизонтали


        # Определяем угол поворота (в градусах) для выравнивания глаз по горизонтали
        angle = math.degrees(math.atan2(dY, dX))

        # Находим центр между глазами (точка вращения)
        eyes_center = (
            int((left_eye[0] + right_eye[0]) // 2),  # Средняя x-координата
            int((left_eye[1] + right_eye[1]) // 2)   # Средняя y-координата
        )

        # Создаём матрицу аффинного поворота вокруг центра глаз
        rotation_matrix = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)

        h, w = image.shape[:2]  # Высота и ширина исходного изображения

        # Применяем поворот к изображению
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

        # Вычисляем расстояние между глазами
        dist = np.sqrt(dX**2 + dY**2)
        # Желаемое расстояние между глазами (40% от ширины выходного изображения)
        desired_dist = desired_face_width * 0.4


        if dist == 0:
            # Если глаза в одной точке — поворот невозможен, используем масштаб 1.0
            print("Предупреждение: dist=0, используем scale=1.0")
            scale = 1.0
            dist_zero = True
        else:
            # Вычисляем масштаб для приведения расстояния между глазами к желаемому
            scale = desired_dist / dist


        # Масштабируем размеры изображения
        new_h, new_w = int(h * scale), int(w * scale)
        # Применяем масштабирование к повёрнутому изображению
        scaled_image = cv2.resize(rotated_image, (new_w, new_h))

        # Обновляем координаты центра глаз после масштабирования
        new_eyes_center = (int(eyes_center[0] * scale), int(eyes_center[1] * scale))


        # Определяем границы ROI (области интереса) вокруг центра глаз
        x1 = max(0, new_eyes_center[0] - desired_face_width // 2)  # Левый край
        y1 = max(0, new_eyes_center[1] - desired_face_width // 2)  # Верхний край
        x2 = x1 + desired_face_width  # Правый край
        y2 = y1 + desired_face_width  # Нижний край

        # Вырезаем выровненное лицо
        aligned_face = scaled_image[y1:y2, x1:x2]

        # Если размер вырезанного лица не совпадает с желаемым — дополнительно масштабируем
        if (aligned_face.shape[0] != desired_face_width or 
            aligned_face.shape[1] != desired_face_width):
            aligned_face = cv2.resize(aligned_face, (desired_face_width, desired_face_width))


        return aligned_face, dist_zero
    