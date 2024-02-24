# meter-identify
manual meter reading is still necessary. In order to achieve automatic recognition of old mechanical water meters, a DL (deep learning) algorithm has been proposed.

本方案将任务拆解为两个子任务：

1、水表读数区域准确估计；

2、对估计出的读数区域中数字准确识别。

## 第一部分：水表读数区域准确估计

### 通过 imgaug.augmenters 进行基础变换，包括尺寸调整、翻转、旋转等

通过 Python 类 `BaseAugment` 的实现，用于图像增强操作，其中主要包括基本的图像处理功能和对应的标注处理。让我们逐步分析这段代码：

1. `__init__` 方法：
   - 初始化函数，用于设置类的属性。接受几个参数：
     - `only_resize`：布尔值，指示是否只进行尺寸调整。
     - `keep_ratio`：布尔值，指示是否保持图像长宽比不变。
     - `augmenters`：图像增强器，使用 imgaug 库中的增强器。
     - `resize_shape`：字典，包含目标尺寸的高度和宽度。
   - 将传入的参数分别赋值给对应的属性。

2. `resize_image` 方法：
   - 接受一个图像作为输入，根据类的属性进行调整大小的操作。
   - 如果设置了保持长宽比，则根据原始图像的长宽比计算新的宽度。
   - 使用 OpenCV 的 `cv2.resize` 方法将图像调整到指定的尺寸。
   - 返回调整大小后的图像。

3. `process` 方法：
   - 接受一个数据字典作为输入，包含待处理的图像以及其他相关信息。
   - 如果设置了增强器，将图像和标注数据应用增强器的变换。如果只进行尺寸调整，则只调用 `resize_image` 方法。
   - 更新数据字典中的图像和相关信息，如文件名和形状。
   - 返回处理后的数据字典。

4. `may_augment_annotation` 方法：
   - 如果给定了增强器，对标注进行相应的变换。
   - 遍历每个线段（可能代表文本行），根据增强器对线段的多边形进行变换。
   - 更新标注数据中的多边形信息，同时考虑是否将该标注视为困难样本（在本例中，如果文本是 '###'，则将其视为困难样本）。
   - 返回更新后的数据字典。

5. `may_augment_poly` 方法：
   - 如果给定了增强器，对多边形进行相应的变换。
   - 将多边形的顶点转换为 `imgaug` 库中的 `Keypoint` 对象。
   - 调用增强器对关键点进行变换。
   - 将变换后的关键点重新转换为多边形的顶点表示。
   - 返回变换后的多边形顶点。

整体上，这段代码实现了一个图像增强的基础框架，包括尺寸调整、图像变换和标注变换等功能，同时支持是否保持长宽比不变以及是否只进行尺寸调整的设置。


请看以下示例代码：

```python
class BaseAugment():
    '''
    通过 imgaug.augmenters 进行基础变换，包括尺寸调整、翻转、旋转等
    '''
    def __init__(self, only_resize=False, keep_ratio=False, augmenters=None, resize_shape=None):
        self.only_resize = only_resize
        self.keep_ratio = keep_ratio
        self.augmenter = augmenters
        self.resize_shape = resize_shape


    def resize_image(self, image):
        origin_height, origin_width, _ = image.shape
        height = self.resize_shape['height']
        width = self.resize_shape['width']
        if self.keep_ratio:    # 是否保持图像长宽比不变
            width = origin_width * height / origin_height
            N = math.ceil(width / 32)
            width = N * 32
        image = cv2.resize(image, (width, height))
        return image

    def process(self, data):
        image = data['image']
        shape = image.shape


        if self.augmenter:
            aug = self.augmenter.to_deterministic()#这是一个图像增强库中的方法，用于将增强器设置为确定性模式，以便在数据增强过程中保持一致性
            if self.only_resize:
                data['image'] = self.resize_image(image)   # 只进行尺寸调整
            else:
                data['image'] = aug.augment_image(image)   # 图像变换
            self.may_augment_annotation(aug, data, shape)  # 对 polygon 标注进行对应的变换


        filename = data.get('filename', data.get('data_id', ''))
        data.update(filename=filename, shape=shape[:2])
        return data

    def may_augment_annotation(self, aug, data, shape):
        if aug is None:
            return data


        line_polys = []
        for line in data['lines']:
            if self.only_resize:
                new_poly = [(p[0], p[1]) for p in line['poly']]
            else:
                new_poly = self.may_augment_poly(aug, shape, line['poly'])
            line_polys.append({
                'points': new_poly,
                'ignore': line['text'] == '###',    # 图像是否是困难样本（模糊不可辨），本任务数据集中不存在困难样本
                'text': line['text'],
            })
        data['polys'] = line_polys
        return data


    def may_augment_poly(self, aug, img_shape, poly):
        keypoints = [imgaug.Keypoint(p[0], p[1]) for p in poly]
        keypoints = aug.augment_keypoints([imgaug.KeypointsOnImage(keypoints, shape=img_shape)])[0].keypoints
        poly = [(p.x, p.y) for p in keypoints]
        return poly
```

### 颜色增强，包括亮度、对比度、饱和度、色相变换

这段代码是一个名为 `ColorJitter` 的 Python 类，用于执行颜色增强操作。以下是对其功能的分析：

1. `__init__` 方法：
   - 初始化函数，用于设置颜色增强的参数。
   - 接受四个参数：`b`（亮度）、`c`（对比度）、`s`（饱和度）、`h`（色相）。
   - 使用 `torchvision.transforms.ColorJitter` 创建颜色增强器，并将参数传递给它。

2. `process` 方法：
   - 接受一个数据字典作为输入，其中包含要处理的图像。
   - 从数据字典中提取图像数据，并将其转换为 PIL 图像对象，然后转换为 RGB 模式。
   - 将 PIL 图像对象传递给颜色增强器进行增强，并将结果转换回 NumPy 数组。
   - 更新数据字典中的图像数据为增强后的图像，并返回数据字典。

整体上，这段代码实现了一个颜色增强类，通过调整图像的亮度、对比度、饱和度和色相来增强图像的视觉效果。









