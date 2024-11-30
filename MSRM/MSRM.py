# 필요 패키지들
import os
import cv2
import numpy as np
import torch
import torchvision.ops as ops
# 
from ultralytics import YOLO

# 이미지 저장
def save(
        image, dir,
        boxes, confidences, classes,
        class_names=None,
        font_scale=0.5, thickness=2
    ):
    # 바운딩 박스가 그려진 이미지 생성
    preview_image = draw_boxes_on_image(
        image, boxes, confidences, classes, class_names, font_scale, thickness
    )
    # 저장
    cv2.imwrite(dir, preview_image)

# 이미지 미리보기
def preview(
        image, 
        boxes, confidences, classes,
        class_names=None,
        window_height=900,
        font_scale=0.5, thickness=2
    ):
    # 창 크기 고정
    cv2.namedWindow('Preview Image', cv2.WINDOW_NORMAL)

    # 창 크기 계산
    height, width, channel = image.shape
    window_h = window_height
    window_w = int(width * (window_h / height))

    # 창 크기 조절
    cv2.resizeWindow('Preview Image', window_w, window_h)

    # 바운딩 박스가 그려진 이미지 생성
    preview_image = draw_boxes_on_image(
        image, boxes, confidences, classes, class_names, font_scale, thickness
    )

    # 새로운 크기로 이미지 리사이즈 (선명도 유지 위해 INTER_AREA 사용)
    preview_image = cv2.resize(preview_image, (window_w, window_h), interpolation=cv2.INTER_AREA)

    # 바운딩 박스가 그려진 이미지 미리보기
    cv2.imshow('Preview Image', preview_image)
    cv2.waitKey()
    cv2.destroyAllWindows()

# 이미지에 바운딩 박스 그리기 (최대 20개 클래스까지 가능)
def draw_boxes_on_image(
        image, 
        boxes, confidences, classes,
        class_names=None,
        font_scale=0.5, thickness=2
    ):
    # 이미지 복사 (원본 이미지를 수정하지 않도록)
    img_copy = image.copy()

    # 클래스 ID와 색상을 매핑하는 딕셔너리
    class_colors = {
        0: (0, 255, 0),         # 클래스  0: 초록색 -> staff
        1: (192, 192, 192),     # 클래스  1: 회색   -> clef
        2: (255, 0, 255),       # 클래스  2: 자홍색 -> key
        3: (255, 255, 0),       # 클래스  3: 청록색 -> measure
        4: (0, 0, 255),         # 클래스  4: 빨간색 -> rest
        5: (128, 128, 0),       # 클래스  5: 올리브 -> time
        6: (255, 0, 0),         # 클래스  6: 파란색 -> note
        7: (75, 0, 130),        # 클래스  7: 인디고         -> accidental
        8: (255, 192, 203),     # 클래스  8: 핑크색         -> articulation
        9: (0, 255, 127),       # 클래스  9: 시안           -> dynamic
        10: (255, 165, 0),      # 클래스 10: 오렌지색       -> glissando
        11: (139, 69, 19),      # 클래스 11: 초콜릿색       -> octave
        12: (0, 0, 128),        # 클래스 12: 어두운 파란색  -> ornament
        13: (0, 128, 128),      # 클래스 13: 어두운 청록색  -> repetition
        14: (0, 255, 255),      # 클래스 14: 노란색         -> tie
        15: (128, 0, 0),        # 클래스 15: 어두운 빨간색
        16: (0, 128, 0),        # 클래스 16: 어두운 초록색
        17: (255, 105, 180),    # 클래스 17: 딸기색
        18: (255, 215, 0),      # 클래스 18: 금색
        19: (255, 20, 147),     # 클래스 19: 딥 핑크
    }

    # 박스, 점수, 클래스 순회하며 표시
    for i in range(len(boxes)):
        # 박스 좌표 (x_min, y_min, x_max, y_max)
        x_min, y_min, x_max, y_max = map(int, boxes[i])

        # 현재 클래스에 대한 색상 추출
        class_id = int(classes[i])
        color = class_colors[class_id]

        # 바운딩 박스 그리기
        cv2.rectangle(img_copy, (x_min, y_min), (x_max, y_max), color, thickness)

        # 클래스 이름과 점수
        class_id = int(classes[i])
        confidence = float(confidences[i])
        label = f'{confidence:.2f}'  # 점수

        # 클래스 이름이 제공되었으면 추가
        if class_names is not None:
            label = f'{class_names[class_id]}: {confidence:.2f}'

        # 텍스트의 위치와 크기 설정
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(img_copy, (x_min, y_min - text_height - baseline), (x_min + text_width, y_min), color, thickness=cv2.FILLED)
        cv2.putText(img_copy, label, (x_min, y_min - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    #
    return img_copy



# 교집합 영역의 크기를 계산해주는 함수
def box_intersection_area(box1, box2):
    # 좌표값 추출
    b1x1 = box1[:, 0:1]
    b1y1 = box1[:, 1:2]
    b1x2 = box1[:, 2:3]
    b1y2 = box1[:, 3:4]
    b2x1 = box2[:, 0:1]
    b2y1 = box2[:, 1:2]
    b2x2 = box2[:, 2:3]
    b2y2 = box2[:, 3:4]

    # 교집합 영역의 너비와 높이 계산
    inner_width = torch.clamp(torch.minimum(b1x2, b2x2) - torch.maximum(b1x1, b2x1), min=0)
    inner_height = torch.clamp(torch.minimum(b1y2, b2y2) - torch.maximum(b1y1, b2y1), min=0)

    # 교집합 넓이 계산
    intersection_area = inner_width * inner_height

    # 교집합 넓이 반환
    return intersection_area

# 박스들의 합집합 좌표를 반환해주는 함수
def union_boxes(boxes):

    # 좌표값 추출
    x1 = boxes[:, 0:1]
    y1 = boxes[:, 1:2]
    x2 = boxes[:, 2:3]
    y2 = boxes[:, 3:4]

    # 합집합 좌표 계산
    x1_min = torch.min(x1, dim=0, keepdim=True)[0]
    y1_min = torch.min(y1, dim=0, keepdim=True)[0]
    x2_max = torch.max(x2, dim=0, keepdim=True)[0]
    y2_max = torch.max(y2, dim=0, keepdim=True)[0]

    # 하나로 연결
    union_box = torch.cat((x1_min, y1_min, x2_max, y2_max), dim=-1)

    # 합집합 좌표 반환
    return union_box

# 확률 값의 평균을 구함
def union_confidences(confidences):
    return torch.mean(confidences, dim=0, keepdim=True)

# 첫번째 요소를 클래스 번호로 지정
def union_classes(classes):
    # TODO: 나중에 필요하다면 가장 면적이 넓은 것을 기준으로 클래스를 정하도록 코드 수정
    #return classes[0:1]
    return classes.mode()[0].unsqueeze(0)

# 두 박스가 교집합이 있으면 이 둘을 합쳐서 하나의 박스로 만들어주는 함수
def merge_bounding_boxes(boxes, confidences, classes):
    # 디바이스 조사
    device = boxes.device

    # 합집합 완성된 좌표들
    merged_boxes = []
    merged_confidences = []
    merged_classes = []

    # 박스를 더이상 합칠 수 있는게 없을 때 까지 반복
    while boxes.shape[0] > 0:

        # 교집합 여부 조사
        mask = (box_intersection_area(boxes[:1], boxes) > 0).reshape(-1)    # 교집합이 있는 것들 마스크
        mask_not = ~mask.reshape(-1)                                        # 교집합이 없는 것들 마스크

        # 교집합 박스 갯수 계산
        count = torch.sum(mask)

        if count == 1:
            # 교집합이 없는 경우 완성으로 빼기
            merged_boxes.append(boxes[:1])
            merged_confidences.append(confidences[:1])
            merged_classes.append(classes[:1])

            # 업데이트
            boxes = boxes[1:]
            confidences = confidences[1:]
            classes = classes[1:]
        else:
            # 박스 병합 작업
            curr_box = union_boxes(boxes[mask])
            curr_conf = union_confidences(confidences[mask])
            curr_class = union_classes(classes[mask])

            # 업데이트
            boxes       = torch.cat((curr_box,   boxes[mask_not]),       dim=0)
            confidences = torch.cat((curr_conf,  confidences[mask_not]), dim=0)
            classes     = torch.cat((curr_class, classes[mask_not]),     dim=0)

    # 하나의 텐서로 만들기
    if merged_boxes:
        merged_boxes        = torch.cat(merged_boxes, dim=0)
        merged_confidences  = torch.cat(merged_confidences, dim=0)
        merged_classes      = torch.cat(merged_classes, dim=0)
    else:
        merged_boxes = torch.tensor([], device=device)
        merged_confidences = torch.tensor([], device=device)
        merged_classes = torch.tensor([], device=device)

    return merged_boxes, merged_confidences, merged_classes



# 악상기호 대분류 모델
class SymbolDetector:
    def __init__(
        self, 
        image_dirs,             # 분석할 이미지 주소들
        note_confidences=0.7,   # 인식 대상의 정확도 임계값
        note_iou_threshold=0.8, # NMS IoU 임계값
    ):
        # 모델 불러오기
        model_dir = os.path.join(os.path.dirname(__file__), 'best.pt')
        self.model = YOLO(model_dir)
        
        # 이미지 분할 정보
        self.split_height = 416 # YOLO 모델 입력 크기 h
        self.split_width = 416  # YOLO 모델 입력 크기 w
        self.overlap = 128      # 겹치는 부분의 크기

        # 객체 탐지 파라미터 저장
        self.note_confidences=note_confidences      # 인식 대상의 정확도 임계값
        self.note_iou_threshold=note_iou_threshold  # NMS IoU 임계값

        # 이미지들 불러오기
        self.images = self.__load_images(image_dirs)

        # 인식한 이미지 갯수 카운트
        self.recognition_count = 0
                
    # 이미지들 불러오기
    def __load_images(self, dirs):
        # 불러온 이미지들 목록
        images = []

        for dir in dirs:
            # 이미지 로드
            image = cv2.imread(dir)
            if image is None:
                print('이미지 "{dir}"를 찾을 수 없거나 불러올 수 없습니다.')
                continue

            # 이미지 색상 반전
            images.append(255 - image)

        # 불러온 이미지들 반환
        return images

    # 하나의 이미지에 대한 예측 수행
    def __predict(self, image, preview, save):
        # 이미지 높이 계산
        height, width, channels = image.shape

        # 검출할 데이터
        boxes = []
        confidences = []
        classes = []

        # 이미지 분할 및 YOLO 적용
        for y in range(0, height, self.split_height - self.overlap):
            for x in range(0, width, self.split_width - self.overlap):
                # 경계 체크 및 패딩 추가
                split_img = np.zeros((self.split_height, self.split_width, 3), dtype=np.uint8)
                img_patch = image[y:y + self.split_height, x:x + self.split_width]

                # 패딩 처리
                split_img[:img_patch.shape[0], :img_patch.shape[1]] = img_patch

                # 이미지의 픽셀값이 전부 0이면 인식 건너뛰기
                if np.all(split_img == 0):
                    continue  # 인식을 건너뜀

                # YOLO 예측
                result = self.model.predict(split_img, conf=self.note_confidences)
                
                # 결과 저장
                for r in result:
                    # 바운딩 박스의 좌표에 x, y 오프셋을 더하여 원본 이미지 좌표로 변환
                    xyxy = r.boxes.xyxy.clone()
                    xyxy[:, [0, 2]] += x
                    xyxy[:, [1, 3]] += y
                    # 텐서 값이 있는 것(유효한 박스들)들만 추가
                    if xyxy.numel() > 0:
                        boxes.append(xyxy)
                        confidences.append(r.boxes.conf)
                        classes.append(r.boxes.cls)
                
                # 인식되는 과정 미리보기
                if preview:
                    # 하나의 텐서로 만들기
                    if boxes:
                        boxes_tmp = torch.cat(boxes)
                        confidences_tmp = torch.cat(confidences)
                        classes_tmp = torch.cat(classes)
                    else:
                        boxes_tmp = torch.tensor([])
                        confidences_tmp = torch.tensor([])
                        classes_tmp = torch.tensor([])
                    # preview 이미지 생성
                    preview_image = draw_boxes_on_image(
                        255 - image,
                        boxes_tmp, confidences_tmp, classes_tmp
                    )
                    # 이미지 크기에 맞춰 윈도우 크기 변경
                    self.__preview_window_resize(preview_image)
                    # 이미지 화질 선명하기 크기 변경
                    preview_image = self.__resize_image(preview_image)
                    # preview 미리보기
                    cv2.imshow('Preview Image', preview_image)
                    cv2.waitKey(1)
                
                # 영상으로 저장
                if save:
                    # 하나의 텐서로 만들기
                    if boxes:
                        boxes_tmp = torch.cat(boxes)
                        confidences_tmp = torch.cat(confidences)
                        classes_tmp = torch.cat(classes)
                    else:
                        boxes_tmp = torch.tensor([])
                        confidences_tmp = torch.tensor([])
                        classes_tmp = torch.tensor([])
                    # preview 이미지 생성
                    preview_image = draw_boxes_on_image(
                        255 - image,
                        boxes_tmp, confidences_tmp, classes_tmp
                    )
                    # 프레임 영상에 저장
                    self.output_video.write(preview_image)
                    self.output_video.write(preview_image)

        # 각 이미지마다 인식된 박스 데이터 결과를 하나의 텐서로 결합
        if boxes:
            boxes = torch.cat(boxes)
            confidences = torch.cat(confidences)
            classes = torch.cat(classes)
        else:
            boxes = torch.tensor([])
            confidences = torch.tensor([])
            classes = torch.tensor([])

        # 인식한 악보 갯수 카운트 증가
        self.recognition_count += 1
        
        # 검출된 데이터들 반환
        return boxes, confidences, classes

    # 여러이미지의 바운딩 박스 통합
    def __assemble(self, boxes, confidences, classes):
        # 최종 결과
        result_boxes = []
        result_confidences = []
        result_classes = []

        # 클래스별로 NMS 적용 및 결과 통합 
        for class_id in classes.unique():
            # 특정 클래스만 필터링
            mask = classes == class_id

            # 박스 좌표, 점수, 클래스 필터링
            masked_boxes = boxes[mask]
            masked_confidences = confidences[mask]
            masked_classes = classes[mask]

            # 0:staff, 1:clef, 2:key, 3:measure, 5:time
            if class_id==0 or class_id==1 or class_id==2 or class_id==3 or class_id==5:
                # 바운딩 박스가 서로 겹치면 합치기
                new_boxes, new_confidences, new_classes = merge_bounding_boxes(
                    masked_boxes, masked_confidences, masked_classes
                )

                # 최종 결과에 남은 박스 추가
                result_boxes.append(new_boxes)
                result_confidences.append(new_confidences)
                result_classes.append(new_classes)

            # 13:repetition
            elif class_id==13:
                # 데이터셋에서 repetition-middle 이미지..? 이거 제외 시키기.
                pass

            # 14:tie, 10:glissando, 11:octave
            elif class_id==14 or class_id==10 or class_id==11:
                # 바운딩 박스의 confidences 가 0.9 이상이때만 통과시키도록
                conf_indices = masked_confidences > 0.8
                masked_boxes = masked_boxes[conf_indices]
                masked_confidences = masked_confidences[conf_indices]
                masked_classes = masked_classes[conf_indices]

                # 바운딩 박스가 서로 겹치면 합치기
                new_boxes, new_confidences, new_classes = merge_bounding_boxes(
                    masked_boxes, masked_confidences, masked_classes
                )

                # 최종 결과에 남은 박스 추가
                result_boxes.append(new_boxes)
                result_confidences.append(new_confidences)
                result_classes.append(new_classes)

            # 4:rest, 6:note, 7:accidental, 8:articulation, 9:dynamic, 12:ornament
            else:
                # NMS 적용
                nms_indices = ops.nms(masked_boxes, masked_confidences, self.note_iou_threshold) # iou_threshold=0.8 이상이면 합침

                # 최종 결과에 남은 박스 추가
                result_boxes.append(masked_boxes[nms_indices])
                result_confidences.append(masked_confidences[nms_indices])
                result_classes.append(masked_classes[nms_indices])

        # 각 클래스마다 추출된 NMS 결과를 하나의 텐서로 결합
        if result_boxes:
            result_boxes = torch.cat(result_boxes)
            result_confidences = torch.cat(result_confidences)
            result_classes = torch.cat(result_classes)
        else:
            result_boxes = torch.tensor([])
            result_confidences = torch.tensor([])
            result_classes = torch.tensor([])

        # 최종 결과 반환
        return result_boxes, result_confidences, result_classes

    # 이미지 크기에서 벗어나는 좌표 처리
    def __inspection(self, image, boxes, confidences, classes):
        # 이미지 높이 계산
        height, width, channels = image.shape

        # 기존 이미지의 너비와 높이 크기를 넘어가는 좌표 클리핑
        boxes[:, 0] = torch.clamp(boxes[:, 0], min=0, max=width)   # x_min
        boxes[:, 1] = torch.clamp(boxes[:, 1], min=0, max=height)  # y_min
        boxes[:, 2] = torch.clamp(boxes[:, 2], min=0, max=width)   # x_max
        boxes[:, 3] = torch.clamp(boxes[:, 3], min=0, max=height)  # y_max

        # 박스의 크기가 0인 것들 제거
        box_w = boxes[:, 2] - boxes[:, 0]
        box_h = boxes[:, 3] - boxes[:, 1]

        # 너비와 높이가 모두 0이 아닌 박스만 필터링
        mask = (box_w > 0) & (box_h > 0)

        # 유효한 박스만 필터링
        boxes = boxes[mask]
        confidences = confidences[mask]
        classes = classes[mask]

        # 결과 반환
        return boxes, confidences, classes

    # 창 크기 계산
    def __preview_window_resize(self, image, window_height=900):
        # 창의 크기 조절
        cv2.namedWindow('Preview Image', cv2.WINDOW_NORMAL)  # 창 크기 고정

        # 창 크기 계산
        height, width, channel = image.shape
        self.window_h = window_height
        self.window_w = int(width * (self.window_h / height))
        cv2.resizeWindow('Preview Image', self.window_w, self.window_h)

    # 이미지 화질 선명하게 크기 변경
    def __resize_image(self, image):
        # 새로운 크기로 이미지 리사이즈 (선명도 유지 위해 INTER_AREA 사용)
        resized_image = cv2.resize(image, (self.window_w, self.window_h), interpolation=cv2.INTER_AREA)
        return resized_image

    # 객체 인식 수행
    def detact(self, preview=False, save=False):
        # 이미지들 에서 탐지된 객체
        self.datas = []

        # 객체 탐지 시작
        for image in self.images:
            # 영상 객체 생성
            if save:
                height, width, channels = image.shape
                fourcc = cv2.VideoWriter_fourcc(*'X264')
                self.output_video = cv2.VideoWriter('video_' + str(self.recognition_count) + '.mp4', fourcc, 60, (width, height))
            # 이미지 미리보기
            if preview:
                self.__preview_window_resize(image) # 윈도우 크기 변경
                cv2.imshow('Preview Image', self.__resize_image(255 - image))
                cv2.waitKey(1)
            # 데이터 검출
            boxes, confidences, classes = self.__predict(image, preview, save)
            # 데이터 결합
            boxes, confidences, classes = self.__assemble(boxes, confidences, classes)
            # 데이터 검사
            boxes, confidences, classes = self.__inspection(image, boxes, confidences, classes)
            # 데이터 저장
            self.datas.append((boxes, confidences, classes))
            # 이미지 미리보기
            if preview:
                # preview 이미지 생성
                preview_image = draw_boxes_on_image(
                    255 - image,
                    boxes, confidences, classes
                )
                # 이미지 크기에 맞춰 윈도우 크기 변경
                self.__preview_window_resize(preview_image)
                # 이미지 화질 선명하기 크기 변경
                preview_image = self.__resize_image(preview_image)
                # preview 미리보기
                cv2.imshow('Preview Image', preview_image)
                cv2.waitKey(1000)
            # 영상으로 저장
            if save:
                # preview 이미지 생성
                preview_image = draw_boxes_on_image(
                    255 - image,
                    boxes, confidences, classes
                )
                # 프레임 영상에 저장
                for i in range(120):
                    self.output_video.write(preview_image)
            # 영상 객체 닫기
            if save:
                self.output_video.release()
        
        # 이미지 미리보기 닫기
        if preview:
            cv2.waitKey(0)  # 키 입력 대기
            cv2.destroyAllWindows()  # 윈도우 닫기

        # 결과 반환
        return [255 - image for image in self.images], self.datas