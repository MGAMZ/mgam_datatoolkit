# 暮光霭明的万能工具包

**Note:** This repo is now only for preview, I am still working on it. I hope it will help more researchers with easy APIs one day. If you are eager to use several functions, I am happy to offer some help through [my Email](mailto:312065559@qq.com).

## 开源协议

本项目采用 **GNU General Public License v3.0** 开源协议。

详细条款请参考 [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html)。

## 简介

一开始只是为了自己开发方便，整理了大多数常用的函数方法在这个工具包中，以后不知道会不会帮到别人呢？

目前有如下几个模块：

- criterion: 定义一些常见的损失函数，其实一般情况下用别家的就可以了，这里只是应付一些特殊的情况。

- dataset: 根据研究需要，定制一些数据集支持算法。希望能够通过这个包，将各式各样的数据集组织成相同的形式。比如OpenMIM的数据集规范。也可以是一些常见的规范。

- deploy: 用于模型部署时使用的一些方法

- io: 用于定义一些通用的医学领域常见的读写函数。

- mm: OpenMIM框架下自定义组件

- models: 一些著名的神经网络

- process: 数据预处理、后处理

- utils: 其他小工具

## Release Note

### 2024.09.19 V1.0.0

将原始的mgamdata和mmaitrox合并兼容，便于后续进一步开发。

### 2024.09.22 V1.0.0

加入git submodule，一键安装常见mm package。

### V1.1.0.241009

同步readme，版本号格式更改。这十几天里加入了很多新的函数和方法，优化了整个项目的可读性和可维护性，但是依旧有很多不足。

### V1.11.15.250502

已包含了对多项相关研究的支持。

### V1.11.19.250509

希望mgamdata这个开源包能作为我继续前行的无限引擎！
凡是打不倒我的，都将让我更加强大。

为FLARE_2023加入了样本筛选机制，非标准标注现在将会被deprecated。
