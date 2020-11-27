# 今更学ぶTensorFlowLite



### 何でアプリ開発者が知ってないといけない?

最近はオンデバイスでAIを実行する機会が多い。
![alt](vision (1).png)


大抵はライブラリを使用することとなるが、独自モデル(モデルというのはAIの脳みそ)のプロダクトも?
モデルのパフォーマンスが悪い場合、アプリとしてもっさりしてしまうこともしばしば・・・。

そのような場合、モデルをモバイルデバイス向けに最適化する手法があるため
アプリエンジニアも知っておいた方がより良い。

---

### そもそもTensorFlowって?

TensorFlowは機械学習向けにGoogleが開発したオープンソースプラットフォーム。
モデル構築(人間で言う脳みそ)、学習、推論(モデルに答えを出してもらうこと)を簡単に行える。
TensorFlowをより使いやすくしたKerasと呼ばれるラッパーも存在する。

大抵のAIプロダクトはTensorFlow、Kerasでの開発となっている。
PyTorchと呼ばれるプラットフォームも最近はアツいが、
モバイル向けに組み込むとなるとGoogle製のTensorFlow、Kerasを選択する場合が多い。

![google-trend-DLframework](/Users/r_shimizu/yumemiapk/tensorflowlite_slide/google-trend-DLframework.png)

---

### じゃあ、TensorFlowLiteって?

TensorFlowを使用して生成したモデルを限られたデバイスリソースでも効率的に実行(推論)出来るように
形式を変換したもの。
Android、iOSなど様々なデバイスで実行可能。

モデル変換を行うと、ファイルサイズの縮小、精度に影響を与えない最適化が導入される。
それにより、モデルロード時間と推論に掛かる時間が短くなる。

---

### ファイルサイズの縮小、最適化って何をしてるの?

何をするかは実際に変換を行うML開発者が決めるが、量子化を行うのが一般的

---

### 量子化って?

モデルが計算時に使用する変数を浮動小数点数からより小さい表現範囲(float16や8bit整数等)の変数に変換すること。
それにより、モデルのサイズと推論に必要な時間を減らすことが出来る。
精度の低下はごくわずか。

---

### 量子化の効果ってどれくらいあるの?

以下の画像はMobileNet V1と呼ばれるアーキテクチャのモデルで比較する
2019年、2020年の量子化前、量子化後のパフォーマンス。

`2019年`
量子化前のモデルをCPUで動かした場合に50ms後半掛かっているものが、
量子化後は30ms後半まで高速化している。
`2020年`
量子化前は30ms後半だったものが、
量子化後に15ms程度にまで高速化している。

MobileNet V1はモバイル向けに考案されたアーキテクチャで、
現在はV3まで存在するため、より高速になっている。

![chart](/Users/r_shimizu/yumemiapk/tensorflowlite_slide/chart.png)

---

### 変換はアプリエンジニアが行うの?

変換はMLエンジニアが行うべきだと個人的には思っている。

量子化にもいくつか選択肢があり、モデルの観点から見た最適な選択、
その選択によって精度への影響がどれくらいあるかの検証作業、
また、変換にはPythonでTensorFlowプラットフォームを使用する為、そのような知見も必要。
![スクリーンショット 2020-11-11 8.51.55](/Users/r_shimizu/Desktop/スクリーンショット 2020-11-11 8.51.55.png)


```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_quantized_model)
```

---

### アプリエンジニアは何するの?

MLエンジニアから提供されたモデルをアプリ上で動作させる



### 何すればいいの?

https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android

上記URLがAndroid上で画像識別モデルを動作させる際のサンプルコード(Java😇)

---

### 重要なところを見ていく

#### TensorFlowライブラリの追加

```
implementation 'org.tensorflow:tensorflow-lite-support:0.0.0-nightly'
```

---

#### モデルファイルとラベルファイルの配置
![スクリーンショット 2020-11-26 22.03.20](/Users/r_shimizu/yumemiapk/tensorflowlite_slide/スクリーンショット 2020-11-26 22.03.20.png)

Mobilenet_v1_1.0_224_quant.tfliteが量子化したモデルファイルで、
labels.txtがラベルファイル(識別したい物が記載されているファイル)となる。

#### モデルファイルの読み込み

```java
MappedByteBuffer tfliteModel = FileUtil.loadMappedFile(activity, "mobilenet_v1_1.0_224_quant.tflite");
```

#### ラベルファイルの読み込み

```java
labels = FileUtil.loadLabels(activity, "labels.txt");
```
---
#### Interpreterの生成

```java
tflite = new Interpreter(tfliteModel, tfliteOptions);
```

Interpreterは実際にモデル推論を実施するクラス。
optionとしてスレッド数の設定などが出来る。

#### 推論の実施

```java
tflite.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());
```

run()で推論を実施できる。
inputImageBufferは学習した際に使用した画像と同じサイズにリサイズした画像のバッファ。TensorImageクラス。
outputProbabilityBufferは結果を出力する用のバッファ。TensorBufferクラス。

---

#### 推論結果を取得

```java
Map<String, Float> labeledProbability = new TensorLabel(labels, probabilityProcessor.process(outputProbabilityBuffer))
            .getMapWithFloatValue();
```

ラベルと確率のマップの変換できる。

---

#### サンプルアプリだとこんな感じ

<img src="/Users/r_shimizu/yumemiapk/tensorflowlite_slide/Screenshot_20201111-095910.png" alt="Screenshot_20201111-095910" style="zoom:25%;" />

---

#### まとめ

個人的にオンデバイスでモデルを動作させるのはAIの知識が0の状態ではまだ簡単では無い。
MLエンジニアとアプリエンジニアの知識の共有がかなり大切な印象。
アプリエンジニアが量子化などの少し踏み込んだ概念だけでも知っていれば連携が上手くいきそう。

