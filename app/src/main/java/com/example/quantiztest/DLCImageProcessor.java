package com.example.quantiztest;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.util.Log;

import com.qualcomm.qti.snpe.FloatTensor;
import com.qualcomm.qti.snpe.NeuralNetwork;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class DLCImageProcessor {
    private static final String TAG = "DLCImageProcessor";
    private static final int INPUT_SIZE = 640; // YOLONas 모델의 입력 크기
    private static final int NUM_DETECTIONS = 8400; // 모델 출력 형상에 맞게 수정 (8400개 탐지)
    private static final int NUM_CLASSES = 80; // COCO 데이터셋 클래스 수

    private NeuralNetwork neuralNetwork;
    private List<String> labels;
    private Context context;
    private String inputTensorName;
    private String boxesTensorName;
    private String scoresTensorName;
    private String classesTensorName;

    public DLCImageProcessor(Context context, NeuralNetwork neuralNetwork) {
        this.context = context;
        this.neuralNetwork = neuralNetwork;

        // 입력 및 출력 텐서 이름 가져오기
        if (neuralNetwork != null) {
            this.inputTensorName = neuralNetwork.getInputTensorsNames().iterator().next();

            // 출력 텐서 이름 매핑 (순서에 따라 다를 수 있음)
            List<String> outputNames = new ArrayList<>(neuralNetwork.getOutputTensorsNames());
            Log.d("dlcCheck", "모든 출력 텐서 이름: " + outputNames);

            // 출력 텐서 이름 매핑 (순서에 따라 다를 수 있으므로 로그로 확인)
            if (outputNames.size() >= 3) {
                this.boxesTensorName = outputNames.get(0);     // boxes 텐서
                this.scoresTensorName = outputNames.get(1);    // scores 텐서
                this.classesTensorName = outputNames.get(2);   // class_idx 텐서

                Log.d("dlcCheck", "사용 출력 텐서 매핑:");
                Log.d("dlcCheck", "박스 텐서: " + boxesTensorName);
                Log.d("dlcCheck", "점수 텐서: " + scoresTensorName);
                Log.d("dlcCheck", "클래스 텐서: " + classesTensorName);
            } else {
                Log.e("dlcCheck", "출력 텐서가 3개 미만입니다. 현재: " + outputNames.size());
            }
        }

        try {
            this.labels = loadLabels();
        } catch (IOException e) {
            Log.e("dlcCheck", "라벨 파일을 로드하는 중 오류 발생: " + e.getMessage());
            this.labels = new ArrayList<>();
        }
    }

    /**
     * assets 폴더에서 labels.txt 파일을 로드합니다.
     */
    private List<String> loadLabels() throws IOException {
        List<String> labels = new ArrayList<>();
        BufferedReader reader = new BufferedReader(
                new InputStreamReader(context.getAssets().open("labels.txt")));
        String line;
        while ((line = reader.readLine()) != null) {
            labels.add(line);
        }
        reader.close();
        Log.d("dlcCheck", "로드된 라벨 수: " + labels.size());
        return labels;
    }

    /**
     * 비트맵 이미지를 처리하고 객체 탐지를 수행합니다.
     * @param bitmap 처리할 이미지
     * @return 탐지된 객체 목록
     */
    public List<YoloImageProcessor.Detection> processImage(Bitmap bitmap) {
        if (neuralNetwork == null || inputTensorName == null) {
            Log.e("dlcCheck", "신경망 또는 입력 텐서 이름이 null입니다.");
            return new ArrayList<>();
        }

        Log.d("dlcCheck", "이미지 처리 시작, 입력 텐서 이름: " + inputTensorName);

        // 입력 이미지 준비
        Bitmap resizedBitmap = resizeBitmap(bitmap, INPUT_SIZE, INPUT_SIZE);

        // RGB 이미지 데이터를 float 배열로 변환
        float[] imgData = new float[INPUT_SIZE * INPUT_SIZE * 3];
        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        resizedBitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

        int imgIdx = 0;
        for (int i = 0; i < INPUT_SIZE * INPUT_SIZE; ++i) {
            int pixel = pixels[i];
            // RGB 채널 추출 및 정규화
            imgData[imgIdx++] = ((pixel >> 16) & 0xFF) / 255.0f; // R
            imgData[imgIdx++] = ((pixel >> 8) & 0xFF) / 255.0f;  // G
            imgData[imgIdx++] = (pixel & 0xFF) / 255.0f;         // B
        }

        try {
            // 입력 텐서 생성
            FloatTensor inputTensor = neuralNetwork.createFloatTensor(
                    neuralNetwork.getInputTensorsShapes().get(inputTensorName)
            );

            // 입력 데이터 설정
            inputTensor.write(imgData, 0, imgData.length);

            // 입력 텐서 맵 생성
            Map<String, FloatTensor> inputs = new HashMap<>();
            inputs.put(inputTensorName, inputTensor);

            long startTime = System.currentTimeMillis();

            // 모델 실행
            Map<String, FloatTensor> outputs = neuralNetwork.execute(inputs);

            long endTime = System.currentTimeMillis();
            Log.d("dlcCheck", "DLC 모델 추론 시간: " + (endTime - startTime) + "ms");

            // 출력 처리
            List<YoloImageProcessor.Detection> detections = new ArrayList<>();
            float confidenceThreshold = 0.4f;

            // 출력 텐서 로깅
            Log.d("dlcCheck", "출력 텐서 조회 시작");
            for (String outputName : outputs.keySet()) {
                FloatTensor tensor = outputs.get(outputName);
                Log.d("dlcCheck", "출력 텐서: " + outputName + ", 크기: " + tensor.getSize());
            }

            // 출력 텐서 확인
            if (!outputs.containsKey(boxesTensorName) ||
                    !outputs.containsKey(scoresTensorName) ||
                    !outputs.containsKey(classesTensorName)) {
                Log.e("dlcCheck", "필요한 출력 텐서가 없습니다.");

                // 텐서 릴리즈
                inputTensor.release();
                return detections;
            }

            // 출력 텐서 가져오기
            FloatTensor boxesTensor = outputs.get(boxesTensorName);
            FloatTensor scoresTensor = outputs.get(scoresTensorName);
            FloatTensor classesTensor = outputs.get(classesTensorName);

            // 출력 텐서 형상 로그
            Log.d("dlcCheck", "박스 텐서 형상: " + shapeDimsToString(boxesTensor.getShape()));
            Log.d("dlcCheck", "점수 텐서 형상: " + shapeDimsToString(scoresTensor.getShape()));
            Log.d("dlcCheck", "클래스 텐서 형상: " + shapeDimsToString(classesTensor.getShape()));

            // 텐서 데이터 가져오기
            float[] boxesData = new float[boxesTensor.getSize()];
            float[] scoresData = new float[scoresTensor.getSize()];
            float[] classesData = new float[classesTensor.getSize()];

            boxesTensor.read(boxesData, 0, boxesData.length);
            scoresTensor.read(scoresData, 0, scoresData.length);
            classesTensor.read(classesData, 0, classesData.length);

            // 모든 탐지 결과를 저장할 리스트
            List<YoloImageProcessor.Detection> allDetections = new ArrayList<>();

            // 탐지 결과 로깅 (처음 몇 개만)
            for (int i = 0; i < Math.min(10, NUM_DETECTIONS); i++) {
                Log.d("dlcCheck", "탐지 #" + i +
                        " - 점수: " + scoresData[i] +
                        ", 클래스: " + (int)classesData[i] +
                        ", 박스: [" +
                        boxesData[i*4] + ", " +
                        boxesData[i*4+1] + ", " +
                        boxesData[i*4+2] + ", " +
                        boxesData[i*4+3] + "]");
            }

            // 각 탐지 결과 처리
            for (int i = 0; i < NUM_DETECTIONS; ++i) {
                // 신뢰도 점수
                float confidence = scoresData[i];
                confidence = Math.min(confidence, 1.0f);

                // 신뢰도 임계값 이상인 결과만 처리
                if (confidence > confidenceThreshold) {
                    // 클래스 인덱스
                    int classIndex = (int) classesData[i];

                    if (classIndex >= 0 && classIndex < labels.size()) {
                        String label = labels.get(classIndex);

                        // 원하는 클래스만 필터링 (person, cup, apple, banana)
                        if (!label.equals("cup") && !label.equals("person") &&
                                !label.equals("apple") && !label.equals("banana")) {
                            continue;
                        }

                        // 박스 좌표 (YOLO 출력은 [x1, y1, x2, y2] 형식)
                        float x1 = boxesData[i * 4];
                        float y1 = boxesData[i * 4 + 1];
                        float x2 = boxesData[i * 4 + 2];
                        float y2 = boxesData[i * 4 + 3];

                        // 정규화 (0~1 범위로)
                        x1 = x1 / INPUT_SIZE;
                        y1 = y1 / INPUT_SIZE;
                        x2 = x2 / INPUT_SIZE;
                        y2 = y2 / INPUT_SIZE;

                        // 원본 이미지 크기에 맞게 변환
                        float left = x1 * bitmap.getWidth();
                        float top = y1 * bitmap.getHeight();
                        float right = x2 * bitmap.getWidth();
                        float bottom = y2 * bitmap.getHeight();

                        // 좌표가 유효한지 확인
                        if (left < 0) left = 0;
                        if (top < 0) top = 0;

                        // 바운딩 박스 크기가 유효한지 확인
                        if (right > left && bottom > top) {
                            YoloImageProcessor.Detection detection = new YoloImageProcessor.Detection(
                                    label, confidence, left, top, right, bottom);
                            allDetections.add(detection);

                            if (allDetections.size() <= 5) {
                                Log.d("dlcCheck", "탐지 결과 #" + allDetections.size() +
                                        " - 라벨: " + label +
                                        ", 신뢰도: " + confidence +
                                        ", 좌표: [" + left + ", " + top + ", " + right + ", " + bottom + "]");
                            }
                        }
                    }
                }
            }

            Log.d("dlcCheck", "총 탐지 수: " + allDetections.size() + ", NMS 적용 전");

            // NMS 적용하여 중복 제거
            List<YoloImageProcessor.Detection> filteredDetections = applyNMS(allDetections, 0.7f);
            Log.d("dlcCheck", "NMS 적용 후 최종 탐지 수: " + filteredDetections.size());

            // 텐서 릴리즈
            inputTensor.release();
            for (FloatTensor tensor : outputs.values()) {
                tensor.release();
            }

            return filteredDetections;

        } catch (Exception e) {
            Log.e("dlcCheck", "DLC 모델 실행 중 오류 발생: " + e.getMessage(), e);
            e.printStackTrace();
            return new ArrayList<>();
        } finally {
            if (resizedBitmap != null && resizedBitmap != bitmap) {
                resizedBitmap.recycle();
            }
        }
    }

    private String shapeDimsToString(int[] dims) {
        if (dims == null) return "null";
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < dims.length; i++) {
            sb.append(dims[i]);
            if (i < dims.length - 1) sb.append(", ");
        }
        sb.append("]");
        return sb.toString();
    }

    private List<YoloImageProcessor.Detection> applyNMS(List<YoloImageProcessor.Detection> detections, float iouThreshold) {
        // 신뢰도가 없으면 빈 리스트 반환
        if (detections.isEmpty()) {
            return new ArrayList<>();
        }

        // 신뢰도 기준으로 내림차순 정렬
        List<YoloImageProcessor.Detection> sortedDetections = new ArrayList<>(detections);
        Collections.sort(sortedDetections, (d1, d2) -> Float.compare(d2.getConfidence(), d1.getConfidence()));

        List<YoloImageProcessor.Detection> selectedDetections = new ArrayList<>();
        boolean[] isRemoved = new boolean[sortedDetections.size()];

        // 사람 클래스에 대해 더 높은 IoU 임계값 사용
        float personIouThreshold = 0.65f;  // 사람 클래스에 대한 더 높은 임계값

        for (int i = 0; i < sortedDetections.size(); i++) {
            if (isRemoved[i]) continue;

            YoloImageProcessor.Detection current = sortedDetections.get(i);
            selectedDetections.add(current);

            for (int j = i + 1; j < sortedDetections.size(); j++) {
                if (isRemoved[j]) continue;

                YoloImageProcessor.Detection next = sortedDetections.get(j);

                // 같은 클래스의 객체만 비교
                if (!current.getLabel().equals(next.getLabel())) {
                    continue;
                }

                // IoU 계산
                float iou = calculateIoU(current, next);

                // 현재 클래스에 맞는 임계값 선택
                float threshold = current.getLabel().equals("person") ? personIouThreshold : iouThreshold;

                // IoU가 임계값보다 크면 중복으로 간주하고 제거
                if (iou > threshold) {
                    isRemoved[j] = true;
                }
            }
        }

        Log.d("dlcCheck", "NMS 적용 전 탐지 수: " + detections.size() + ", 적용 후: " + selectedDetections.size());
        return selectedDetections;
    }

    /**
     * 두 바운딩 박스 간의 IoU(Intersection over Union)를 계산합니다.
     */
    private float calculateIoU(YoloImageProcessor.Detection d1, YoloImageProcessor.Detection d2) {
        // 겹치는 영역 계산
        float xLeft = Math.max(d1.getLeft(), d2.getLeft());
        float yTop = Math.max(d1.getTop(), d2.getTop());
        float xRight = Math.min(d1.getRight(), d2.getRight());
        float yBottom = Math.min(d1.getBottom(), d2.getBottom());

        // 겹치는 영역이 없으면 0 반환
        if (xRight < xLeft || yBottom < yTop) return 0;

        float intersectionArea = (xRight - xLeft) * (yBottom - yTop);

        // 각 바운딩 박스의 면적 계산
        float d1Area = (d1.getRight() - d1.getLeft()) * (d1.getBottom() - d1.getTop());
        float d2Area = (d2.getRight() - d2.getLeft()) * (d2.getBottom() - d2.getTop());

        // IoU 계산
        return intersectionArea / (d1Area + d2Area - intersectionArea);
    }

    /**
     * 비트맵을 지정된 크기로 리사이즈합니다.
     */
    private Bitmap resizeBitmap(Bitmap bitmap, int width, int height) {
        float scaleWidth = ((float) width) / bitmap.getWidth();
        float scaleHeight = ((float) height) / bitmap.getHeight();
        Matrix matrix = new Matrix();
        matrix.postScale(scaleWidth, scaleHeight);
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, false);
    }
}