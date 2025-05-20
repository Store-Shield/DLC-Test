package com.example.quantiztest;

import android.app.Application;
import android.content.Context;
import android.content.res.AssetManager;
import android.os.Build;
import android.util.Log;

import com.qualcomm.qti.snpe.NeuralNetwork;
import com.qualcomm.qti.snpe.SNPE;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.Date;
import java.util.Set;

public class DLCLoader {
    private static final String TAG = "DLCLoader";
    private String modelName;
    private NeuralNetwork neuralNetwork;
    private Context context;

    public DLCLoader(Context context, String modelName) {
        this.context = context;
        this.modelName = modelName;
    }

    // DLCLoader.java에 새로운 메서드 추가
    private void collectSNPEDiagnosticInfo() {
        try {
            Application app = (Application) context.getApplicationContext();

            // 1. SNPE 버전 확인
            try {
                String snpeVersion = SNPE.getRuntimeVersion(app);
                Log.d(TAG, "SNPE 런타임 버전: " + snpeVersion);
            } catch (Exception e) {
                Log.e(TAG, "SNPE 런타임 버전 확인 실패: " + e.getMessage());
            }

            // 2. 네이티브 라이브러리 확인
            String nativeLibDir = context.getApplicationInfo().nativeLibraryDir;
            Log.d(TAG, "네이티브 라이브러리 디렉토리: " + nativeLibDir);

            File libDir = new File(nativeLibDir);
            if (libDir.exists() && libDir.isDirectory()) {
                File[] files = libDir.listFiles();
                if (files != null) {
                    Log.d(TAG, "네이티브 라이브러리 파일 개수: " + files.length);
                    for (File file : files) {
                        Log.d(TAG, "네이티브 라이브러리: " + file.getName() + " (" + file.length() + " 바이트)");
                    }
                } else {
                    Log.d(TAG, "네이티브 라이브러리 목록을 가져오지 못했거나 디렉토리가 비어있습니다");
                }
            } else {
                Log.d(TAG, "네이티브 라이브러리 디렉토리가 존재하지 않거나 디렉토리가 아닙니다");
            }

            // 3. 디바이스 정보 확인
            String manufacturer = Build.MANUFACTURER;
            String model = Build.MODEL;
            String product = Build.PRODUCT;
            String device = Build.DEVICE;
            String hardware = Build.HARDWARE;
            String soc = "";

            // SoC 정보 추출 시도
            try {
                Process process = Runtime.getRuntime().exec("getprop ro.board.platform");
                BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
                soc = reader.readLine();
                reader.close();
            } catch (Exception e) {
                Log.e(TAG, "SoC 정보 확인 중 오류 발생: " + e.getMessage());
            }

            Log.d(TAG, "디바이스 정보 - 제조사: " + manufacturer);
            Log.d(TAG, "디바이스 정보 - 모델: " + model);
            Log.d(TAG, "디바이스 정보 - 제품명: " + product);
            Log.d(TAG, "디바이스 정보 - 기기명: " + device);
            Log.d(TAG, "디바이스 정보 - 하드웨어: " + hardware);
            Log.d(TAG, "디바이스 정보 - SoC: " + soc);

            // 4. DLC 파일 확인
            File dlcFile = new File(context.getFilesDir(), modelName);
            if (dlcFile.exists()) {
                Log.d(TAG, "DLC 파일 - 경로: " + dlcFile.getAbsolutePath());
                Log.d(TAG, "DLC 파일 - 크기: " + dlcFile.length() + " 바이트");
                Log.d(TAG, "DLC 파일 - 마지막 수정일: " + new Date(dlcFile.lastModified()));

                // 파일 헤더 확인 (첫 16바이트)
                try {
                    FileInputStream fis = new FileInputStream(dlcFile);
                    byte[] header = new byte[16];
                    int bytesRead = fis.read(header);
                    fis.close();

                    StringBuilder sb = new StringBuilder();
                    sb.append("DLC 파일 - 헤더: ");
                    for (int i = 0; i < bytesRead; i++) {
                        sb.append(String.format("%02X ", header[i]));
                    }
                    Log.d(TAG, sb.toString());
                } catch (Exception e) {
                    Log.e(TAG, "DLC 파일 헤더 읽기 오류: " + e.getMessage());
                }
            } else {
                Log.d(TAG, "DLC 파일이 존재하지 않습니다: " + dlcFile.getAbsolutePath());
            }

            // 5. SNPE 런타임 지원 여부 테스트 (다양한 방법으로)
            try {
                SNPE.NeuralNetworkBuilder builder = new SNPE.NeuralNetworkBuilder(app);

                // CPU 지원 확인
                boolean cpuSupported = builder.isRuntimeSupported(NeuralNetwork.Runtime.CPU);
                Log.d(TAG, "CPU 런타임 지원 여부: " + cpuSupported);

                // GPU 지원 확인
                boolean gpuSupported = builder.isRuntimeSupported(NeuralNetwork.Runtime.GPU);
                Log.d(TAG, "GPU 런타임 지원 여부: " + gpuSupported);

                // DSP 지원 확인
                boolean dspSupported = builder.isRuntimeSupported(NeuralNetwork.Runtime.DSP);
                Log.d(TAG, "DSP 런타임 지원 여부: " + dspSupported);

                try {
                    // GPU_FLOAT16 지원 확인
                    boolean gpuFloat16Supported = builder.isRuntimeSupported(NeuralNetwork.Runtime.GPU_FLOAT16);
                    Log.d(TAG, "GPU_FLOAT16 런타임 지원 여부: " + gpuFloat16Supported);
                } catch (Exception e) {
                    Log.e(TAG, "GPU_FLOAT16 지원 확인 중 오류: " + e.getMessage());
                }

                // 다른 RuntimeCheckOption으로 다시 시도
                try {
                    builder.setRuntimeCheckOption(NeuralNetwork.RuntimeCheckOption.BASIC_CHECK);
                    Log.d(TAG, "BASIC_CHECK 옵션으로 다시 확인:");

                    cpuSupported = builder.isRuntimeSupported(NeuralNetwork.Runtime.CPU);
                    Log.d(TAG, "CPU 런타임 지원 여부 (BASIC_CHECK): " + cpuSupported);

                    gpuSupported = builder.isRuntimeSupported(NeuralNetwork.Runtime.GPU);
                    Log.d(TAG, "GPU 런타임 지원 여부 (BASIC_CHECK): " + gpuSupported);

                    dspSupported = builder.isRuntimeSupported(NeuralNetwork.Runtime.DSP);
                    Log.d(TAG, "DSP 런타임 지원 여부 (BASIC_CHECK): " + dspSupported);
                } catch (Exception e) {
                    Log.e(TAG, "BASIC_CHECK 옵션 적용 중 오류: " + e.getMessage());
                }
            } catch (Exception e) {
                Log.e(TAG, "런타임 지원 확인 중 오류 발생: " + e.getMessage());
            }

        } catch (Exception e) {
            Log.e(TAG, "SNPE 진단 정보 수집 중 오류 발생: " + e.getMessage(), e);
        }
    }

    /**
     * Assets 폴더에서 DLC 모델을 로드합니다.
     */
    public boolean loadModelFromAssets() {
        Log.d(TAG, "Loading DLC model from assets: " + modelName);
        try {
            File modelFile = copyAssetToStorage(context, modelName);

            if (modelFile == null || !modelFile.exists()) {
                Log.e(TAG, "Model file does not exist: " + modelName);
                return false;
            }

            // Application 객체 얻기
            Application app = (Application) context.getApplicationContext();

            // NeuralNetworkBuilder 생성
            SNPE.NeuralNetworkBuilder builder = new SNPE.NeuralNetworkBuilder(app);

            // 사용 가능한 런타임 확인
            boolean cpuSupported = builder.isRuntimeSupported(NeuralNetwork.Runtime.CPU);
            boolean gpuSupported = builder.isRuntimeSupported(NeuralNetwork.Runtime.GPU);
            boolean dspSupported = builder.isRuntimeSupported(NeuralNetwork.Runtime.DSP);

            Log.d(TAG, "Runtime support check - CPU: " + cpuSupported +
                    ", GPU: " + gpuSupported +
                    ", DSP: " + dspSupported);

            // 지원되는 런타임만 사용하도록 설정
            if (gpuSupported || dspSupported) {
                // 지원되는 런타임을 동적으로 결정
                int supportedCount = 0;
                if (gpuSupported) supportedCount++;
                if (dspSupported) supportedCount++;
                if (cpuSupported) supportedCount++;

                NeuralNetwork.Runtime[] runtimes = new NeuralNetwork.Runtime[supportedCount];
                int index = 0;

                if (gpuSupported) {
                    runtimes[index++] = NeuralNetwork.Runtime.GPU;
                    Log.d(TAG, "Adding GPU to runtime order");
                }

                if (dspSupported) {
                    runtimes[index++] = NeuralNetwork.Runtime.DSP;
                    Log.d(TAG, "Adding DSP to runtime order");
                }

                if (cpuSupported) {
                    runtimes[index++] = NeuralNetwork.Runtime.CPU;
                    Log.d(TAG, "Adding CPU to runtime order");
                }

                if (supportedCount > 0) {
                    builder.setRuntimeOrder(runtimes);
                    Log.d(TAG, "Set " + supportedCount + " supported runtimes");
                } else {
                    // 실패 시 기본 CPU 설정 - setRuntime이 없으므로 setRuntimeOrder로 대체
                    builder.setRuntimeOrder(new NeuralNetwork.Runtime[]{NeuralNetwork.Runtime.CPU});
                    Log.d(TAG, "No supported runtimes found, using CPU only");
                }
            } else {
                // CPU만 사용 - setRuntime이 없으므로 setRuntimeOrder로 대체
                builder.setRuntimeOrder(new NeuralNetwork.Runtime[]{NeuralNetwork.Runtime.CPU});
                Log.d(TAG, "Only CPU runtime is used (GPU and DSP not supported)");
            }

            // 모델 파일 설정
            builder.setModel(modelFile);
            Log.d(TAG, "Model file set: " + modelFile.getAbsolutePath());

            // CPU 폴백 활성화
            builder.setCpuFallbackEnabled(true);
            Log.d(TAG, "CPU fallback enabled");

            // 성능 프로파일 설정
            builder.setPerformanceProfile(NeuralNetwork.PerformanceProfile.HIGH_PERFORMANCE);
            Log.d(TAG, "Performance profile set to HIGH_PERFORMANCE");

            // DSP 버퍼 설정 (지원되는 경우만)
            if (dspSupported) {
                builder.setUseUserSuppliedBuffers(true);
                Log.d(TAG, "User supplied buffers enabled for DSP");
            }

            // 디버그 모드 활성화
            builder.setDebugEnabled(true);
            Log.d(TAG, "Debug mode enabled");

            // 모델 로드 시도
            try {
                Log.d(TAG, "Attempting to build neural network...");
                neuralNetwork = builder.build();
                Log.d(TAG, "Neural network built successfully");
            } catch (Exception e) {
                Log.e(TAG, "Failed to build neural network: " + e.getMessage());

                // CPU만 사용하는 간단한 구성으로 다시 시도
                Log.d(TAG, "Retrying with CPU-only configuration...");
                builder = new SNPE.NeuralNetworkBuilder(app);
                // setRuntime이 없으므로 setRuntimeOrder로 대체
                builder.setRuntimeOrder(new NeuralNetwork.Runtime[]{NeuralNetwork.Runtime.CPU});
                builder.setModel(modelFile);

                try {
                    neuralNetwork = builder.build();
                    Log.d(TAG, "Neural network built successfully with CPU-only configuration");
                } catch (Exception e2) {
                    Log.e(TAG, "Failed to build neural network with CPU-only: " + e2.getMessage());
                    throw e2;
                }
            }

            // 모델 정보 출력
            if (neuralNetwork != null) {
                Set<String> inputNames = neuralNetwork.getInputTensorsNames();
                Set<String> outputNames = neuralNetwork.getOutputTensorsNames();

                Log.d(TAG, "DLC model loaded successfully: " + modelName);
                Log.d(TAG, "Used runtime: " + neuralNetwork.getRuntime());
                Log.d(TAG, "Input tensor names: " + inputNames);
                Log.d(TAG, "Output tensor names: " + outputNames);

                // 텐서 형태 정보 출력
                for (String inputName : inputNames) {
                    int[] shape = neuralNetwork.getInputTensorsShapes().get(inputName);
                    if (shape != null) {
                        Log.d(TAG, "Input tensor '" + inputName + "' shape: " +
                                java.util.Arrays.toString(shape));
                    }
                }

                for (String outputName : outputNames) {
                    int[] shape = neuralNetwork.getOutputTensorsShapes().get(outputName);
                    if (shape != null) {
                        Log.d(TAG, "Output tensor '" + outputName + "' shape: " +
                                java.util.Arrays.toString(shape));
                    }
                }

                return true;
            } else {
                Log.e(TAG, "Neural network is null after build attempts");
                return false;
            }
        } catch (Exception e) {
            Log.e(TAG, "Error loading DLC model: " + e.getMessage(), e);
            e.printStackTrace();
            return false;
        }
    }
    /**
     * Assets에서 모델 파일을 내부 저장소로 복사합니다.
     */
    private File copyAssetToStorage(Context context, String assetName) throws IOException {
        AssetManager assetManager = context.getAssets();
        InputStream is = assetManager.open(assetName);

        // 내부 저장소에 파일 생성
        File outputFile = new File(context.getFilesDir(), assetName);

        // 이미 존재하는 경우 그대로 반환
        if (outputFile.exists()) {
            Log.d(TAG, "Using existing model file: " + outputFile.getAbsolutePath());
            return outputFile;
        }

        // 파일 복사
        OutputStream os = new FileOutputStream(outputFile);
        byte[] buffer = new byte[1024];
        int read;
        while ((read = is.read(buffer)) != -1) {
            os.write(buffer, 0, read);
        }
        os.flush();
        os.close();
        is.close();

        Log.d(TAG, "Model file copied to: " + outputFile.getAbsolutePath() + " Size: " + outputFile.length() + " bytes");
        return outputFile;
    }

    /**
     * SNPE 신경망 인스턴스를 반환합니다.
     */
    public NeuralNetwork getNeuralNetwork() {
        return neuralNetwork;
    }

    /**
     * 리소스를 해제합니다.
     */
    public void close() {
        if (neuralNetwork != null) {
            Log.d(TAG, "Closing DLC neural network");
            neuralNetwork.release();
            neuralNetwork = null;
        }
    }
}