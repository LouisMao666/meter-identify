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

```python
class ColorJitter():
    '''
    颜色增强，包括亮度、对比度、饱和度、色相变换
    '''
    def __init__(self, b=0.2, c=0.2, s=0.15, h=0.15):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=b, contrast=c, saturation=s, hue=h)
    def process(self, data):
        img = data['image']
        image = Image.fromarray(img.astype('uint8')).convert('RGB')   # 数据类型转换
        img = np.array(self.color_jitter(image)).astype(np.float64)
        data['image'] = img
        return data
```

### 随机裁剪图像，并保证裁剪时不切割到图像中的文字区域

这段代码是一个用于随机裁剪图像并确保裁剪时不会切割到图像中的文字区域的类 `RandomCropData`。让我们逐步分析这段代码：

1. `__init__` 方法：
   - 初始化函数，用于设置随机裁剪的参数。
   - 接受一个参数 `size`，表示裁剪后的图像大小，默认为 (640, 640)。
   - 设置了两个属性 `size`、`max_tries` 和 `min_crop_side_ratio`，分别表示裁剪尝试的最大次数和裁剪区域边长最小比例。

2. `process` 方法：
   - 接受一个数据字典作为输入，其中包含待处理的图像和多边形标注。
   - 提取图像和标注数据，并进行裁剪操作。
   - 将裁剪后的图像和更新后的标注数据存储在数据字典中，并返回该数据字典。

3. `is_poly_outside_rect` 方法：
   - 判断多边形是否在给定的矩形区域外。

4. `split_regions` 方法：
   - 将一维数组中的连续区域分割出来。

5. `random_select` 方法：
   - 从一维数组中随机选择两个值作为切割线的位置。

6. `region_wise_random_select` 方法：
   - 从多个连续区域中随机选择两个区域，并分别从这两个区域中选择一个值作为切割线的位置。

7. `crop_area` 方法：
   - 根据给定的图像和多边形标注，确定裁剪区域。
   - 首先计算文本区域的边界，然后根据设定的最小裁剪比例和最大尝试次数，随机选择裁剪区域。
   - 确保裁剪区域中至少包含一个文字区域，最终返回裁剪区域的位置和尺寸。

整体上，这段代码实现了一个随机裁剪图像的功能，并且通过一系列方法来保证裁剪时不会切割到图像中的文字区域，以确保数据的完整性和准确性。

```python
class RandomCropData():
    '''
    随机裁剪图像，并保证裁剪时不切割到图像中的文字区域
    '''
    def __init__(self, size=(640, 640)):
        self.size = size
        self.max_tries = 10             # 裁剪尝试的最大次数（因为存在裁剪区域太小等裁剪失败情况）
        self.min_crop_side_ratio = 0.1  # 裁剪区域边长最小比例，即裁剪的图像边长与原始图像边长的比值不能小于 min_crop_side_ratio

    def process(self, data):
        img = data['image']


        ori_img = img
        ori_lines = data['polys']
        all_care_polys = [line['points'] for line in data['polys'] if not line['ignore']]
        crop_x, crop_y, crop_w, crop_h = self.crop_area(img, all_care_polys)   # 裁剪区域的左上角坐标(x, y)以及区域宽高(w, h)

        # 根据裁剪区域参数对图像进行裁剪，并填充空白以得到指定 size 的图像（在右侧或者底侧进行填充）
        scale_w = self.size[0] / crop_w
        scale_h = self.size[1] / crop_h
        scale = min(scale_w, scale_h)
        h = int(crop_h * scale)
        w = int(crop_w * scale)
        padimg = np.zeros((self.size[1], self.size[0], img.shape[2]), img.dtype)
        padimg[:h, :w] = cv2.resize(img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], (w, h))
        img = padimg

        # 根据裁剪区域参数对文字位置坐标进行转换
        lines = []
        for line in data['polys']:
            poly = ((np.array(line['points']) - (crop_x, crop_y)) * scale).tolist()
            if not self.is_poly_outside_rect(poly, 0, 0, w, h): lines.append({**line, 'points': poly}) # 不保留裁剪区域之外的文字

        data['polys'] = lines
        data['image'] = img


        return data




    def is_poly_outside_rect(self, poly, x, y, w, h):
        # 判断文字polygon 是否在矩形区域外
        poly = np.array(poly)
        if poly[:, 0].max() < x or poly[:, 0].min() > x + w:
            return True
        if poly[:, 1].max() < y or poly[:, 1].min() > y + h:
            return True
        return False


    def split_regions(self, axis):
        # 返回可划切割线的连续区域
        regions = []
        min_axis = 0
        for i in range(1, axis.shape[0]):
            if axis[i] != axis[i-1] + 1:
                region = axis[min_axis:i]
                min_axis = i
                regions.append(region)
        return regions


    def random_select(self, axis, max_size):
        # 从一块连续区域中选择两条切割线
        xx = np.random.choice(axis, size=2)
        xmin = np.min(xx)
        xmax = np.max(xx)
        xmin = np.clip(xmin, 0, max_size - 1)
        xmax = np.clip(xmax, 0, max_size - 1)
        return xmin, xmax


    def region_wise_random_select(self, regions, max_size):
        # 从两块连续区域中选择两条切割线
        selected_index = list(np.random.choice(len(regions), 2))
        selected_values = []
        for index in selected_index:
            axis = regions[index]
            xx = int(np.random.choice(axis, size=1))
            selected_values.append(xx)
        xmin = min(selected_values)
        xmax = max(selected_values)
        return xmin, xmax


    def crop_area(self, img, polys):
        # 裁剪区域
        h, w, _ = img.shape
        h_array = np.zeros(h, dtype=np.int32)
        w_array = np.zeros(w, dtype=np.int32)
        for points in polys:
            points = np.round(points, decimals=0).astype(np.int32)
            minx = np.min(points[:, 0])
            maxx = np.max(points[:, 0])
            w_array[minx:maxx] = 1
            miny = np.min(points[:, 1])
            maxy = np.max(points[:, 1])
            h_array[miny:maxy] = 1
        # h_array == 1 的位置表示有文本，h_array == 0 的位置表示无文本；w_array 同理
        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]


        if len(h_axis) == 0 or len(w_axis) == 0:
            return 0, 0, w, h


        h_regions = self.split_regions(h_axis)
        w_regions = self.split_regions(w_axis)


        for i in range(self.max_tries):
            if len(w_regions) > 1:
                # 有多块可切割区域时
                xmin, xmax = self.region_wise_random_select(w_regions, w)
            else:
                # 只有一块可切割区域时
                xmin, xmax = self.random_select(w_axis, w)
            if len(h_regions) > 1:
                ymin, ymax = self.region_wise_random_select(h_regions, h)
            else:
                ymin, ymax = self.random_select(h_axis, h)


            if xmax - xmin < self.min_crop_side_ratio * w or ymax - ymin < self.min_crop_side_ratio * h:
                # 切割区域太小，不可取
                continue
            num_poly_in_rect = 0

            # 保证至少有一个文字区域在切割出的区域中即可
            for poly in polys:
                if not self.is_poly_outside_rect(poly, xmin, ymin, xmax - xmin, ymax - ymin):
                    num_poly_in_rect += 1
                    break


            if num_poly_in_rect > 0:
                return xmin, ymin, xmax - xmin, ymax - ymin


        return 0, 0, w, h
```









