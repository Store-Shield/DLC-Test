package com.example.quantiztest;

import android.app.Application;
import android.content.Context;
import android.content.res.AssetManager;
import android.util.Log;

import com.qualcomm.qti.snpe.NeuralNetwork;
import com.qualcomm.qti.snpe.SNPE;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.HashSet;
import java.util.Set;

public class DLCLoader {
    private static final String TAG = "DLCLoader";
    private String modelName;
    private NeuralNetwork neuralNetwork;
    private Application application;

    public DLCLoader(Context context, String modelName) {
        this.application = (Application) context.getApplicationContext();
        this.modelName = modelName;
        Log.d("dlcCheck", "DLCLoader 생성자 호출됨: " + modelName);
    }

    /**
     * Assets 폴더에서 DLC 모델을 로드합니다.
     */
    public boolean loadModelFromAssets() {
        Log.d("dlcCheck", "DLC 모델 로드 시작: " + modelName);
        try {
            // 임시 파일로 모델 복사
            File modelFile = copyModelToCache();

            // 런타임 우선순위 설정 (GPU > DSP > CPU 순으로 선호)
            NeuralNetwork.Runtime[] runtimeOrder = new NeuralNetwork.Runtime[3];
            runtimeOrder[0] = NeuralNetwork.Runtime.GPU;
            runtimeOrder[1] = NeuralNetwork.Runtime.DSP;
            runtimeOrder[2] = NeuralNetwork.Runtime.CPU;

            // 사용 가능한 런타임 확인
            Set<NeuralNetwork.Runtime> availableRuntimes = new HashSet<>();
            for (NeuralNetwork.Runtime runtime : runtimeOrder) {
                boolean isAvailable = isRuntimeAvailable(runtime);
                if (isAvailable) {
                    availableRuntimes.add(runtime);
                    Log.d("dlcCheck", "런타임 " + runtime + " 사용 가능");
                } else {
                    Log.d("dlcCheck", "런타임 " + runtime + " 사용 불가");
                }
            }

            // SNPE 네트워크 생성
            SNPE.NeuralNetworkBuilder builder = new SNPE.NeuralNetworkBuilder(application)
                    .setRuntimeOrder(runtimeOrder)
                    .setModel(modelFile)
                    .setCpuFallbackEnabled(true)
                    .setUseUserSuppliedBuffers(false);

            neuralNetwork = builder.build();

            // 모델 정보 출력
            Log.d("dlcCheck", "DLC 모델 로드 성공: " + modelName);
            Log.d("dlcCheck", "사용 중인 런타임: " + neuralNetwork.getRuntime());
            Log.d("dlcCheck", "입력 텐서 이름: " + neuralNetwork.getInputTensorsNames());
            Log.d("dlcCheck", "출력 텐서 이름: " + neuralNetwork.getOutputTensorsNames());

            // 각 입력 텐서 정보 출력
            for (String inputName : neuralNetwork.getInputTensorsNames()) {
                int[] dims = neuralNetwork.getInputTensorsShapes().get(inputName);
                Log.d("dlcCheck", "입력 텐서 [" + inputName + "] 형상: " + shapeDimsToString(dims));
            }

            // 각 출력 텐서 정보 출력
            for (String outputName : neuralNetwork.getOutputTensorsNames()) {
                int[] dims = neuralNetwork.getOutputTensorsShapes().get(outputName);
                Log.d("dlcCheck", "출력 텐서 [" + outputName + "] 형상: " + shapeDimsToString(dims));
            }

            return true;
        } catch (Exception e) {
            Log.e("dlcCheck", "DLC 모델 로드 실패: " + e.getMessage(), e);
            return false;
        }
    }

    private boolean isRuntimeAvailable(NeuralNetwork.Runtime runtime) {
        try {
            // NeuralNetwork.RuntimeCheckOption을 사용하여 런타임 가용성 확인
            SNPE.NeuralNetworkBuilder builder = new SNPE.NeuralNetworkBuilder(application);
            return builder.isRuntimeSupported(runtime);
        } catch (Exception e) {
            Log.e("dlcCheck", "런타임 확인 중 오류 발생: " + e.getMessage());
            return false;
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

    /**
     * Assets에서 모델 파일을 캐시 디렉토리로 복사합니다.
     */
    private File copyModelToCache() throws IOException {
        AssetManager assetManager = application.getAssets();
        InputStream is = assetManager.open(modelName);

        // 임시 파일 생성
        File tempFile = new File(application.getCacheDir(), modelName);

        // 이미 존재하면 삭제
        if (tempFile.exists()) {
            tempFile.delete();
        }

        // 임시 파일에 assets 내용 복사
        copyInputStreamToFile(is, tempFile);

        // 리소스 정리
        is.close();

        Log.d("dlcCheck", "모델 파일 복사 완료: " + tempFile.getAbsolutePath() + ", 크기: " + tempFile.length() + " 바이트");
        return tempFile;
    }

    /**
     * InputStream의 내용을 File로 복사합니다.
     */
    private void copyInputStreamToFile(InputStream in, File file) throws IOException {
        OutputStream out = new FileOutputStream(file);
        byte[] buffer = new byte[1024];
        int read;
        while ((read = in.read(buffer)) != -1) {
            out.write(buffer, 0, read);
        }
        out.flush();
        out.close();
    }

    /**
     * DLC 신경망을 반환합니다.
     */
    public NeuralNetwork getNeuralNetwork() {
        return neuralNetwork;
    }

    /**
     * 리소스를 해제합니다.
     */
    public void close() {
        if (neuralNetwork != null) {
            Log.d("dlcCheck", "DLC 신경망 리소스 해제");
            neuralNetwork.release();
            neuralNetwork = null;
        }
    }
}