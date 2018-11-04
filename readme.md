pytorch版本的CTC实现，参考espnet及mxnet：example/speech recognition实现，基于python2和pytorch0.4.1，使用json格式数据，cfgfile记录配置文件。后期计划延拓到S2S，joint attention、RNN.T。以及Phone、grapheme、word和BPE
​    egs下面放各个子实验的脚本，数据使用kaldi预处理，整理成json格式数据以供拓展
​    src下放主代码





本文件用于迭代版本更新过程： 

#### TODO：

- 初始先在TIMIT下搭建好实验框架
  - Data loader:

    - [ ] matplot画acc图、loss图
    - [ ] adam等有lr的optimizer，lr进行annealing
    - [ ] save checkpoint
    - [ ] predict代码分离，无test label时出预测的结果
    - [ ] deepspeech类似：buckets形式，每个epoch在bucket内shuffle
    - [ ] 一开始epoch升/降序，后面random
    - [ ] espnet类似的dataloader
    - [ ] label没必要也padding到batch的最大，可以设置一个max_bucket_key?
    - [ ] 传到github上，在集群上从github load代码

  - multi gpu实验

  - CTC 模型(BLSTMP、BSLSTM etc，可调节参数)

  - tensorboard 使用
- swbd上CTC先跑通流程，开始搭建s2s
- kaldi ark 转化为hdf5， like espnet
- 注意跟进一下espnet、returnn这些的进展，看那边有没有什么trik
#### FINISHED：

- 初始TIMIT实验

  - 把kaldi格式数据整理成json格式 2018.10.19
  - adadelta:

    - Best eval total: 26.14

    - best dev per24.17->26.57

    - best dev loss->26.83
  - sgd: 32.23
  - Adam:1e-3
  - 之前mxnet：train_fc4  e95: dev per ： 26.04 -- > eval : 25.2(batch norm, normalizer, gaussian_noise, bs=12, adam)

目前最大的问题是为什么pytorch版本的dev per更好，但是eval per却更差

Adadelta, bs30, epoch 100: 26.14

Adadelta, bs12, epoch 100: 25.12(cer: 25.53, loss: 26.40)

Adam 1e-3, bs30, epoch 100: 26.46

Sgd,1e-5, bs30, epoch 100:







#### details

- BLSTM最后先做了一个2hidden->hidden的映射，再做hidden->output的映射。本次CTCArch实现也采用这种方法(考虑BLSTM输出为hidden_dim)，再在最后一层做输出层；而TIMITArch则直接将2hidden->output