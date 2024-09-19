# 暮光霭明的万能工具包

## 开源协议

本项目采用 **GNU General Public License v3.0** 开源协议。

*版权所有 (c) 2024 暮光霭明*

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
