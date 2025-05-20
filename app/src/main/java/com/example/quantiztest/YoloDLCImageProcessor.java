package com.example.quantiztest;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import com.qualcomm.qti.snpe.FloatTensor;
import com.qualcomm.qti.snpe.NeuralNetwork;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class YoloDLCImageProcessor {
    private static final String TAG = "YoloDLCProcessor";
    private Context context;
    private NeuralNetwork neuralNetwork;
    private List<String> labels;

    // 입력 이미지 크기
    private static final int INPUT_SIZE = 640;

    // 허용되는 객체 클래스 목록
    private static final List<String> ALLOWED_CLASSES = List.of("cup", "person", "apple", "banana");

    public YoloDLCImageProcessor(Context context, NeuralNetwork neuralNetwork) {
        this.context = context;
        this.neuralNetwork = neuralNetwork;
        try {
            this.labels = loadLabels();
        } catch (IOException e) {
            Log.e(TAG, "라벨 파일을 로드하는 중 오류 발생: " + e.getMessage());
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
        Log.d(TAG, "로드된 라벨 수: " + labels.size());
        return labels;
    }

    /**
     * 비트맵 이미지를 처리하고 객체 탐지를 수행합니다.
     * SimpleTracker에 직접 입력 가능한 Detection 객체 목록을 반환합니다.
     */
    public List<YoloImageProcessor.Detection> processImage(Bitmap bitmap) {
        try {
            // 1. 모델 정보 로깅
            Log.d(TAG, "DLC 모델 정보 확인:");
            Set<String> inputNames = neuralNetwork.getInputTensorsNames();
            Log.d(TAG, "입력 텐서 수: " + inputNames.size());
            Log.d(TAG, "입력 텐서 이름: " + inputNames);

            Set<String> outputNames = neuralNetwork.getOutputTensorsNames();
            Log.d(TAG, "출력 텐서 수: " + outputNames.size());
            Log.d(TAG, "출력 텐서 이름: " + outputNames);

            // 2. 입력 이미지 준비
            Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true);

            // 3. 입력 텐서 생성
            long startTime = System.currentTimeMillis();
            Map<String, FloatTensor> inputTensorsMap = new HashMap<>();

            for (String inputName : inputNames) {
                // 모델의 입력 형태에 맞는 텐서 생성
                int[] inputShape = neuralNetwork.getInputTensorsShapes().get(inputName);
                if (inputShape != null) {
                    Log.d(TAG, "Creating input tensor for: " + inputName + " with shape: " +
                            java.util.Arrays.toString(inputShape));

                    // 입력 텐서 생성
                    FloatTensor inputTensor = createInputTensor(resizedBitmap, inputShape);
                    inputTensorsMap.put(inputName, inputTensor);
                }
            }

            if (inputTensorsMap.isEmpty()) {
                throw new RuntimeException("모델 입력 텐서를 생성할 수 없습니다.");
            }

            Log.d(TAG, "입력 텐서 생성 시간: " + (System.currentTimeMillis() - startTime) + "ms");

            // 4. 모델 실행
            Log.d(TAG, "DLC 모델 실행 시작...");
            long inferenceStart = System.currentTimeMillis();

            // 모델 실행 및 출력 텐서 얻기
            Map<String, FloatTensor> outputTensorsMap = neuralNetwork.execute(inputTensorsMap);

            long inferenceEnd = System.currentTimeMillis();
            Log.d(TAG, "DLC 모델 추론 시간: " + (inferenceEnd - inferenceStart) + "ms");

            // 5. 출력 텐서 정보 로깅
            for (String name : outputTensorsMap.keySet()) {
                FloatTensor tensor = outputTensorsMap.get(name);

                if (tensor != null) {
                    Log.d(TAG, "출력 텐서 이름: " + name);
                    Log.d(TAG, "출력 텐서 크기: " + tensor.getSize());

                    // 텐서 형태 출력
                    int[] shape = tensor.getShape();
                    Log.d(TAG, "출력 텐서 형상: " + java.util.Arrays.toString(shape));
                }
            }

            // 6. 결과 처리
            List<YoloImageProcessor.Detection> detections = postProcessYoloOutputs(
                    outputTensorsMap, bitmap.getWidth(), bitmap.getHeight());

            // 7. 리소스 해제
            for (FloatTensor tensor : inputTensorsMap.values()) {
                tensor.release();
            }
            for (FloatTensor tensor : outputTensorsMap.values()) {
                tensor.release();
            }

            return detections;

        } catch (Exception e) {
            Log.e(TAG, "이미지 처리 중 오류 발생: " + e.getMessage(), e);
            e.printStackTrace();
            return new ArrayList<>();
        }
    }

    /**
     * 비트맵 이미지로부터 입력 텐서를 생성합니다.
     */
    private FloatTensor createInputTensor(Bitmap bitmap, int[] shape) {
        // 입력 형태 확인
        if (shape.length != 4) {
            throw new IllegalArgumentException("입력 텐서는 4차원이어야 합니다: [batch, height, width, channels]");
        }

        int batch = shape[0];
        int height = shape[1];
        int width = shape[2];
        int channels = shape[3];

        // 입력 텐서 생성
        FloatTensor inputTensor = neuralNetwork.createFloatTensor(shape);

        // 비트맵 픽셀을 float 배열로 변환
        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        // 텐서에 데이터 설정 (NHWC 형식: [batch, height, width, channels])
        float[] tensorData = new float[batch * height * width * channels];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pixel = pixels[y * width + x];

                // RGB 채널 추출 (0-255)
                float r = ((pixel >> 16) & 0xFF);
                float g = ((pixel >> 8) & 0xFF);
                float b = (pixel & 0xFF);

                // NHWC 형식에 맞게 인덱스 계산 (배치 크기 1 가정)
                int idx = (y * width + x) * channels;

                // 정규화 [0, 1] 범위로 변환
                tensorData[idx] = r / 255.0f;
                tensorData[idx + 1] = g / 255.0f;
                tensorData[idx + 2] = b / 255.0f;
            }
        }

        // 텐서에 데이터 설정
        inputTensor.write(tensorData, 0, tensorData.length);

        return inputTensor;
    }

    /**
     * YOLO 모델의 출력을 후처리하여 탐지 객체 목록으로 변환합니다.
     */
    private List<YoloImageProcessor.Detection> postProcessYoloOutputs(
            Map<String, FloatTensor> outputTensorsMap, int originalWidth, int originalHeight) {

        List<YoloImageProcessor.Detection> detections = new ArrayList<>();
        float confidenceThreshold = 0.4f;

        try {
            // 출력 텐서 확인 및 처리
            FloatTensor boxesTensor = null;
            FloatTensor scoresTensor = null;
            FloatTensor classesTensor = null;

            // 출력 텐서 이름과 형태로 적절한 텐서 찾기
            for (String name : outputTensorsMap.keySet()) {
                FloatTensor tensor = outputTensorsMap.get(name);
                int[] shape = tensor.getShape();

                if (shape.length == 3 && shape[2] == 4) {
                    // [batch, num_detections, 4] 형태는 박스 좌표일 가능성 높음
                    boxesTensor = tensor;
                    Log.d(TAG, "박스 텐서로 인식: " + name);
                } else if (shape.length == 2) {
                    // 2차원 텐서는 scores나 classes일 가능성 높음
                    if (scoresTensor == null) {
                        scoresTensor = tensor;
                        Log.d(TAG, "신뢰도 텐서로 인식: " + name);
                    } else {
                        classesTensor = tensor;
                        Log.d(TAG, "클래스 텐서로 인식: " + name);
                    }
                }
            }

            // 필요한 텐서가 모두 있는지 확인
            if (boxesTensor == null || scoresTensor == null) {
                Log.e(TAG, "필요한 출력 텐서를 찾을 수 없습니다.");
                return detections;
            }

            // 텐서 데이터 읽기
            int[] boxShape = boxesTensor.getShape();
            int numDetections = boxShape[1]; // 두 번째 차원이 탐지 수

            // 박스 데이터 읽기 [batch, num_detections, 4]
            float[] boxesData = new float[boxesTensor.getSize()];
            boxesTensor.read(boxesData, 0, boxesData.length);

            // 신뢰도 점수 읽기 [batch, num_detections]
            float[] scoresData = new float[scoresTensor.getSize()];
            scoresTensor.read(scoresData, 0, scoresData.length);

            // 클래스 인덱스 읽기 [batch, num_detections] (있는 경우)
            float[] classesData = null;
            if (classesTensor != null) {
                classesData = new float[classesTensor.getSize()];
                classesTensor.read(classesData, 0, classesData.length);
            }

            // 탐지 결과 처리
            for (int i = 0; i < numDetections; i++) {
                float confidence = scoresData[i];

                // 신뢰도 임계값 이상인 결과만 처리
                if (confidence > confidenceThreshold) {
                    // 클래스 인덱스 가져오기
                    int classIndex;
                    if (classesData != null) {
                        classIndex = (int) classesData[i];
                    } else {
                        // 클래스 텐서가 없는 경우, 박스와 스코어만 있는 모델로 간주
                        classIndex = 0;
                    }

                    // 인덱스가 유효한지 확인
                    if (classIndex >= 0 && classIndex < labels.size()) {
                        String label = labels.get(classIndex);

                        // 허용된 클래스만 처리
                        if (ALLOWED_CLASSES.contains(label)) {
                            // 박스 좌표 인덱스 (NHWC 형식: [batch, num_detections, 4])
                            int boxIdx = i * 4;

                            // 박스 좌표 읽기 (형식: x1, y1, x2, y2 또는 cx, cy, w, h)
                            float x1 = boxesData[boxIdx];
                            float y1 = boxesData[boxIdx + 1];
                            float x2 = boxesData[boxIdx + 2];
                            float y2 = boxesData[boxIdx + 3];

                            // 좌표가 정규화되어 있는 경우 원본 이미지 크기로 변환
                            float left = x1 * originalWidth;
                            float top = y1 * originalHeight;
                            float right = x2 * originalWidth;
                            float bottom = y2 * originalHeight;

                            // 좌표가 유효한지 확인
                            if (left < 0) left = 0;
                            if (top < 0) top = 0;
                            if (right > originalWidth) right = originalWidth;
                            if (bottom > originalHeight) bottom = originalHeight;

                            // postProcessYoloOutputs 메서드 내의 수정된 부분
                        // 바운딩 박스 크기가 유효한지 확인
                            if (right > left && bottom > top) {
                                // YoloImageProcessor.Detection 객체 생성 (SimpleTracker가 기대하는 타입)
                                YoloImageProcessor.Detection detection = new YoloImageProcessor.Detection(
                                        label, confidence, left, top, right, bottom);
                                detections.add(detection);

                                Log.d(TAG, "탐지: " + label +
                                        ", 신뢰도: " + confidence +
                                        ", 좌표: " + left + "," + top + "," + right + "," + bottom);
                            }
                        }
                    }
                }
            }

        } catch (Exception e) {
            Log.e(TAG, "결과 처리 중 오류 발생: " + e.getMessage(), e);
            e.printStackTrace();
        }

        return detections;
    }

    /**
     * 기존 YoloImageProcessor.Detection 클래스와 동일한 구조로 유지
     * SimpleTracker가 이 포맷을 기대하므로
     */
    public static class Detection {
        private final String label;
        private final float confidence;
        private final float left;
        private final float top;
        private final float right;
        private final float bottom;

        public Detection(String label, float confidence, float left, float top, float right, float bottom) {
            this.label = label;
            this.confidence = confidence;
            this.left = left;
            this.top = top;
            this.right = right;
            this.bottom = bottom;
        }

        public String getLabel() {
            return label;
        }

        public float getConfidence() {
            return confidence;
        }

        public float getLeft() {
            return left;
        }

        public float getTop() {
            return top;
        }

        public float getRight() {
            return right;
        }

        public float getBottom() {
            return bottom;
        }

        @Override
        public String toString() {
            return label + " (" + String.format("%.2f", confidence * 100) + "%)";
        }
    }
}