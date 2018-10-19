pytorch版本的CTC实现，参考espnet及mxnet：example/speech recognition实现，基于python2和pytorch0.4.1，使用json格式数据，cfgfile记录配置文件。后期计划延拓到S2S，joint attention、RNN.T。以及Phone、grapheme、word和BPE
    egs下面放各个子实验的脚本，数据使用kaldi预处理，整理成json格式数据以供拓展
    src下放主代码





本文件用于迭代版本更新过程：

#### TODO：

- 初始先在TIMIT下搭建好实验框架
  - 把kaldi格式数据整理成json格式
  - 打印所有配置信息

#### FINISHED：

