# analysis/analyzer.py

from pathlib import Path
from typing import Tuple, Dict, List, Any, Optional
import numpy as np
import cv2
import config
from preprocessing.contour_detector import detect_contour
from preprocessing.dense_contour import get_aligned_dense_contour
from measurement.geometry import align_contour
from measurement.measuring_size import (
    measure_gabarits,
    measure_bottom_diameter,
    measure_straightness_deviation
)
from measurement.orientation import determine_neck_orientation
from utils import save_result_image
from visualization.contour_alignment_viz import visualize_contour_alignment
from visualization.annotated_result_viz import visualize_dimensioned_drawing
from visualization.straightness_viz import visualize_straightness
from loguru import logger


class PreformGeometryAnalyzer:
    def __init__(self, logger=None, pixels_per_mm: float = 14.2, dimensions_data: Dict[str, Any] = None):
        self.logger = logger or logger
        self.pixels_per_mm = pixels_per_mm
        self.dimensions_data = dimensions_data or {}

    def analyze(self, img: np.ndarray, image_id: str = "unknown") -> Tuple[str, Dict, List]:
        try:
            original_img = img.copy()

            # --- ШАГ 1: Обнаружение контура ---
            approx, mask, selected_contour = detect_contour(img, image_id=image_id)

            if approx is None:
                self.logger.warning(
                    f"[{image_id}] Преформа отбракована: не найден валидный контур "
                    f"(возможно, касается края изображения, слишком мала, не вытянута или не обнаружена)."
                )

                rejection_img = original_img.copy()
                text = "REJECTED: part of the preform outside the image"
                cv2.putText(rejection_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

                name = Path(image_id).stem
                output_path = config.DEBUG_OUTPUT_FOLDER / f"{name}_REJECTED.jpg"
                cv2.imwrite(str(output_path), rejection_img)

                return "Отбраковано", {}, []

            # --- ШАГ 2: Выравнивание контура ---
            aligned_approx, angle, cx, cy, rot_mat = align_contour(approx, mask)
            if aligned_approx.size == 0:
                raise ValueError("Выровненный контур пуст.")

            # --- ШАГ 3: (резерв) ---

            # --- ШАГ 4: Измерение габаритов (L, Z) ---
            gabarit_data_before = measure_gabarits(aligned_approx, pixels_per_mm=self.pixels_per_mm)
            extreme_before = gabarit_data_before["extreme_points"]

            # --- ШАГ 5: Определение ориентации ---
            M = cv2.moments(aligned_approx)
            if M["m00"] == 0:
                cx_orient = float(np.mean([p[0] for p in aligned_approx[:, 0, :]]))
                cy_orient = float(np.mean([p[1] for p in aligned_approx[:, 0, :]]))
            else:
                cx_orient = M["m10"] / M["m00"]
                cy_orient = M["m01"] / M["m00"]

            neck_orientation = determine_neck_orientation(
                extreme_points=extreme_before,
                cx=cx_orient,
                cy=cy_orient,
                image_id=image_id
            )

            # --- ШАГ 6: Финальные габариты ---
            gabarit_data = gabarit_data_before
            dimensions_px = gabarit_data["dimensions_px"]
            dimensions_mm = gabarit_data.get("dimensions_mm", {})

            # --- ШАГ 7: Плотный выровненный контур ---
            aligned_original_contour: Optional[np.ndarray] = None
            if rot_mat is not None:
                aligned_original_contour = get_aligned_dense_contour(selected_contour, rot_mat, flip_mat=None)

            # --- ШАГ 8: Диаметр дна (d1) ---
            bottom_data = None
            d1_px = 0.0
            d1_mm = 0.0
            if aligned_original_contour is not None and aligned_original_contour.size > 0:
                try:
                    bottom_data = measure_bottom_diameter(
                        aligned_original_contour,
                        pixels_per_mm=self.pixels_per_mm,
                        orientation=neck_orientation
                    )
                    d1_px = bottom_data["dimensions_px"]["d1_px"]
                    d1_mm = bottom_data.get("dimensions_mm", {}).get("d1_mm", d1_px / self.pixels_per_mm)
                except Exception as e:
                    self.logger.warning(f"[{image_id}] Не удалось измерить диаметр дна: {e}")
            else:
                self.logger.warning(f"[{image_id}] Плотный контур недоступен для измерения дна.")

            # --- ШАГ 9: (заглушка) ---

            # --- ШАГ 10: ИЗМЕРЕНИЕ ПРЯМОЛИНЕЙНОСТИ ---
            straightness_deviation_mm = 0.0
            straightness_data = None
            if aligned_original_contour is not None and aligned_original_contour.size > 0:
                try:
                    straightness_data = measure_straightness_deviation(
                        aligned_original_contour,
                        pixels_per_mm=self.pixels_per_mm,
                        neck_percentage=0.15
                    )
                    straightness_deviation_mm = straightness_data["straightness_deviation_mm"]
                except Exception as e:
                    self.logger.warning(f"[{image_id}] Не удалось измерить прямолинейность: {e}")
            else:
                self.logger.warning(f"[{image_id}] Плотный контур недоступен для измерения прямолинейности.")


            # --- ШАГ 11: (заглушка) ---
            neck_data = None
            I_px = I_mm = 0.0
            defects = []

            # --- ВИЗУАЛИЗАЦИЯ НА РАЗНЫХ ЭТАПАХ ---
            if rot_mat is not None:
                if config.STEP_2_ALIGN_CONTOUR:
                    visualize_contour_alignment(
                        original_img=original_img,
                        selected_contour=selected_contour,
                        aligned_approx=aligned_approx,
                        cx=cx_orient,
                        cy=cy_orient,
                        rot_mat=rot_mat,
                        angle=angle,
                        image_id=image_id,
                        dimensions_data=self.dimensions_data,
                        pixels_per_mm=self.pixels_per_mm,
                        step_flag=True
                    )

                if config.STEP_3_MEASURE_DIMENSIONS:
                    visualize_dimensioned_drawing(
                        original_img=original_img,
                        selected_contour=selected_contour,
                        cx=cx_orient,
                        cy=cy_orient,
                        angle=angle,
                        rot_mat=rot_mat,
                        image_id=image_id,
                        dimensions_data=self.dimensions_data,
                        pixels_per_mm=self.pixels_per_mm,
                        gabarit_data=gabarit_data,
                        neck_data=neck_data,
                        bottom_data=bottom_data,
                        neck_orientation=neck_orientation,
                        step_flag=True
                    )

                if config.STEP_10_STRAIGHTNESS and straightness_data is not None:
                    visualize_straightness(
                        original_img=original_img,
                        selected_contour=selected_contour,
                        straightness_data=straightness_data,
                        rot_mat=rot_mat,
                        image_id=image_id,
                        pixels_per_mm=self.pixels_per_mm,
                        dimensions_data=self.dimensions_data
                    )
            else:
                self.logger.warning(f"[{image_id}] Пропуск визуализации: матрица поворота отсутствует.")

            # --- ФИНАЛЬНЫЙ РЕЗУЛЬТАТ ---
            measurements = {
                "L_px": dimensions_px["L_px"],
                "Z_px": dimensions_px["Z_px"],
                "L_mm": dimensions_mm.get("L_mm", dimensions_px["L_px"] / self.pixels_per_mm),
                "Z_mm": dimensions_mm.get("Z_mm", dimensions_px["Z_px"] / self.pixels_per_mm),
                "d1_px": d1_px,
                "d1_mm": d1_mm,
                "I_px": I_px,
                "I_mm": I_mm,
                "straightness_deviation_mm": straightness_deviation_mm,
            }

            verdict = "В процессе"
            self.logger.info(f"Обработка {image_id}: {verdict}")
            self.logger.info(
                f"Измерения: L={measurements['L_mm']:.2f} mm, "
                f"Z={measurements['Z_mm']:.2f} mm, "
                f"d1={measurements['d1_mm']:.2f} mm, "
                f"S={measurements['straightness_deviation_mm']:.3f} mm"
            )

            name = Path(image_id).stem
            output_path = config.OUTPUT_FOLDER / f"{name}_result.jpg"
            save_result_image(original_img, [aligned_approx], aligned_approx, defects, output_path)

            return verdict, measurements, defects

        except Exception as e:
            self.logger.error(f"Ошибка при обработке {image_id}: {e}", exc_info=True)

            error_img = img.copy()
            cv2.putText(error_img, "rejected: part of the preform outside the image", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            name = Path(image_id).stem
            output_path = config.OUTPUT_FOLDER / f"{name}_result.jpg"
            cv2.imwrite(str(output_path), error_img)

            return "Ошибка", {}, []