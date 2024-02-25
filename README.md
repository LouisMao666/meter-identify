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

### 构造文本区域二值图（DB论文中的 probability map），以及用于计算loss的mask

这段代码是一个名为 `MakeSegDetectionData` 的类，用于构造文本区域的二值图（DB 论文中的 probability map）以及用于计算损失的 mask。让我们逐步分析这段代码：

1. `__init__` 方法：
   - 初始化函数，用于设置构造二值图的参数。
   - 接受两个参数 `min_text_size` 和 `shrink_ratio`，分别表示文本区域的最小尺寸和收缩比例。
   - 将这些参数存储为类属性。

2. `process` 方法：
   - 接受一个数据字典作为输入，其中包含图像、多边形标注等信息。
   - 调整数据结构，使其更方便后续操作。
   - 遍历每个多边形标注，根据标注的类型（忽略标签或文本区域尺寸）进行处理，更新对应的 mask。
   - 对于不需要忽略的文本区域，进行多边形的收缩操作，并绘制出二值图（probability map）。
   - 更新数据字典中的信息，包括图像、多边形、二值图和 mask，然后返回该数据字典。

整体上，这段代码实现了根据文本区域标注构造出二值图和 mask 的功能，其中利用了多边形的收缩操作来生成二值图，同时根据文本区域的尺寸和忽略标签来更新 mask。

```python
class MakeSegDetectionData():
    '''
    构造文本区域二值图（DB论文中的 probability map），以及用于计算loss的mask
    '''
    def __init__(self, min_text_size=8, shrink_ratio=0.4):
        self.min_text_size = min_text_size
        self.shrink_ratio = shrink_ratio      # polygon 收缩比例


    def process(self, data):
        # 数据结构调整统一，方便后续操作
        polygons = []
        ignore_tags = []
        annotations = data['polys']
        for annotation in annotations:
            polygons.append(np.array(annotation['points']))
            ignore_tags.append(annotation['ignore'])
        ignore_tags = np.array(ignore_tags, dtype=np.uint8)
        filename = data.get('filename', data['data_id'])
        shape = np.array(data['shape'])
        data = OrderedDict(image=data['image'],
                           polygons=polygons,
                           ignore_tags=ignore_tags,
                           shape=shape,
                           filename=filename,
                           is_training=data['is_training'])

        image = data['image']
        polygons = data['polygons']
        ignore_tags = data['ignore_tags']
        image = data['image']
        filename = data['filename']


        h, w = image.shape[:2]
        if data['is_training']:
            polygons, ignore_tags = self.validate_polygons(polygons, ignore_tags, h, w)
        gt = np.zeros((h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)
        for i in range(len(polygons)):
            polygon = polygons[i]
            height = max(polygon[:, 1]) - min(polygon[:, 1])
            width = max(polygon[:, 0]) - min(polygon[:, 0])
            if ignore_tags[i] or min(height, width) < self.min_text_size: # 文本区域太小时，作为困难样本 ignore
                cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
                ignore_tags[i] = True
            else:
                # 收缩 polygon 并绘制 probability map
                polygon_shape = Polygon(polygon)
                distance = polygon_shape.area * (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
                subject = [tuple(l) for l in polygons[i]]
                padding = pyclipper.PyclipperOffset()
                padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                shrinked = padding.Execute(-distance)
                if shrinked == []:
                    cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
                    ignore_tags[i] = True
                    continue
                shrinked = np.array(shrinked[0]).reshape(-1, 2)
                cv2.fillPoly(gt, [shrinked.astype(np.int32)], 1)


        if filename is None:
            filename = ''
        data.update(image=image,
                    polygons=polygons,
                    gt=gt, mask=mask, filename=filename)
        return data
```

1. `__init__` 方法：
   - 初始化函数，用于设置构造二值图的参数。
   - 接受三个参数 `shrink_ratio`、`thresh_min` 和 `thresh_max`，分别表示收缩比例、二值图中边界的最小值和最大值。
   - 忽略了警告以简化输出。

2. `process` 方法：
   - 接受一个数据字典作为输入，其中包含图像和多边形标注等信息。
   - 遍历每个多边形标注，根据标注的类型进行处理，更新二值图和 mask。
   - 将二值图缩放到指定的范围内，然后将二值图和 mask 存储在数据字典中，并返回该数据字典。

3. `draw_border_map` 方法：
   - 绘制文本边界的二值图。
   - 首先根据收缩比例对多边形进行收缩操作，并绘制出收缩后的多边形，同时更新 mask。
   - 计算多边形的边界，并根据边界计算出边界与像素点之间的距离，然后根据距离更新二值图。

4. `distance` 方法：
   - 计算图像中的点到多边形边界的距离。
   - 接受坐标点和多边形的两个端点作为输入，计算点到边界的距离，并返回结果。

整体上，这段代码实现了根据文本多边形标注构造出文本边界的二值图和 mask 的功能，其中利用了多边形的收缩操作来生成边界二值图，并根据多边形的边界距离计算出边界像素的值。

```python
class MakeBorderMap():
    '''
    构造文本边界二值图（DB论文中的 threshold map），以及用于计算loss的mask
    '''
    def __init__(self, shrink_ratio=0.4, thresh_min=0.3, thresh_max=0.7):
        self.shrink_ratio = shrink_ratio
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max
        warnings.simplefilter("ignore")


    def process(self, data):
        image = data['image']
        polygons = data['polygons']
        ignore_tags = data['ignore_tags']
        canvas = np.zeros(image.shape[:2], dtype=np.float32)
        mask = np.zeros(image.shape[:2], dtype=np.float32)


        for i in range(len(polygons)):
            if ignore_tags[i]:
                continue
            self.draw_border_map(polygons[i], canvas, mask=mask)    # 绘制 border map
        canvas = canvas * (self.thresh_max - self.thresh_min) + self.thresh_min
        data['thresh_map'] = canvas
        data['thresh_mask'] = mask
        return data


    def draw_border_map(self, polygon, canvas, mask):
        polygon = np.array(polygon)
        assert polygon.ndim == 2
        assert polygon.shape[1] == 2


        polygon_shape = Polygon(polygon)
        distance = polygon_shape.area * (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
        subject = [tuple(l) for l in polygon]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        padded_polygon = np.array(padding.Execute(distance)[0])
        cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)


        xmin = padded_polygon[:, 0].min()
        xmax = padded_polygon[:, 0].max()
        ymin = padded_polygon[:, 1].min()
        ymax = padded_polygon[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1


        polygon[:, 0] = polygon[:, 0] - xmin
        polygon[:, 1] = polygon[:, 1] - ymin


        xs = np.broadcast_to(np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
        ys = np.broadcast_to(np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))


        distance_map = np.zeros((polygon.shape[0], height, width), dtype=np.float32)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]
            absolute_distance = self.distance(xs, ys, polygon[i], polygon[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        distance_map = distance_map.min(axis=0)


        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
            1 - distance_map[
                ymin_valid-ymin:ymax_valid-ymax+height,
                xmin_valid-xmin:xmax_valid-xmax+width],
            canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])


    def distance(self, xs, ys, point_1, point_2):
        # 计算图像中的点到 文字polygon 边界的距离
        height, width = xs.shape[:2]
        square_distance_1 = np.square(
            xs - point_1[0]) + np.square(ys - point_1[1])
        square_distance_2 = np.square(
            xs - point_2[0]) + np.square(ys - point_2[1])
        square_distance = np.square(
            point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])


        cosin = (square_distance - square_distance_1 - square_distance_2) / (2 * np.sqrt(square_distance_1 * square_distance_2))
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)
        result = np.sqrt(square_distance_1 * square_distance_2 * square_sin / square_distance)


        result[cosin < 0] = np.sqrt(np.fmin(square_distance_1, square_distance_2))[cosin < 0]
        return result
```

### 将图像元素值归一化到[-1, 1]

这段代码是一个名为 `NormalizeImage` 的类，用于将图像的元素值归一化到 `[-1, 1]` 的范围内。让我们逐步分析这段代码：

1. 类属性 `RGB_MEAN`：
   - 一个长度为 3 的 NumPy 数组，表示 RGB 图像的均值。用于归一化图像。

2. `process` 方法：
   - 接受一个数据字典作为输入，其中包含图像等信息。
   - 从数据字典中提取图像，并进行归一化操作。首先减去 RGB 均值，然后除以 255。
   - 将归一化后的图像转换为 PyTorch 张量，并按照通道顺序重新排列为 `(C, H, W)` 的形状。
   - 更新数据字典中的图像，并返回该数据字典。

3. `restore` 方法（类方法）：
   - 接受一个 PyTorch 张量作为输入，表示需要恢复的图像。
   - 将 PyTorch 张量转换为 NumPy 数组，并进行逆归一化操作。首先将通道重新排列为 `(H, W, C)` 的形状，然后乘以 255，并加上 RGB 均值。
   - 将恢复后的图像转换为 `uint8` 类型，并返回该图像。

整体上，这段代码实现了图像的归一化和逆归一化操作，其中利用了 RGB 均值来归一化图像，并在逆归一化过程中恢复图像的原始值。

```python
class NormalizeImage():
    '''
    将图像元素值归一化到[-1, 1]
    '''
    RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])


    def process(self, data):
        assert 'image' in data, '`image` in data is required by this process'
        image = data['image']
        image -= self.RGB_MEAN
        image /= 255.
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        data['image'] = image
        return data

    @classmethod
    def restore(self, image):
        image = image.permute(1, 2, 0).to('cpu').numpy()
        image = image * 255.
        image += self.RGB_MEAN
        image = image.astype(np.uint8)
        return image
```

###   过滤掉后续不需要的键值对

这段代码是一个名为 `FilterKeys` 的类，用于过滤掉数据字典中不需要的键值对。让我们逐步分析这段代码：

1. `__init__` 方法：
   - 初始化函数，用于设置需要过滤掉的键的集合。
   - 接受一个参数 `superfluous`，表示后续不需要的键值对的键集合。
   - 将传入的键集合转换为集合类型并存储在 `self.superfluous_keys` 中。

2. `process` 方法：
   - 接受一个数据字典作为输入，其中包含待处理的键值对。
   - 遍历 `self.superfluous_keys` 中的键，在数据字典中删除这些键值对。
   - 返回处理后的数据字典。

这段代码实现了一个简单的过滤器，用于从数据字典中删除指定的键值对，以便后续处理过程不需要处理这些键值对。

```python
class FilterKeys():
    '''
    过滤掉后续不需要的键值对
    '''
    def __init__(self, superfluous):
        self.superfluous_keys = set(superfluous)

    def process(self, data):
        for key in self.superfluous_keys:
            del data[key]
        return data
```

### 训练集数据处理

这段代码定义了一个名为 `train_processes` 的列表，其中包含了一系列数据处理步骤，用于训练过程中对数据进行预处理。让我们逐步分析这些处理步骤：

1. `BaseAugment`:
   - 使用基本的数据增强方法，包括水平翻转、旋转和尺寸调整。

2. `ColorJitter`:
   - 进行颜色增强，包括亮度、对比度、饱和度和色相变换。

3. `RandomCropData`:
   - 对图像进行随机裁剪，并保证裁剪时不切割到文本区域。

4. `MakeSegDetectionData`:
   - 构造文本区域二值图（probability map）以及用于计算损失的 mask。

5. `MakeBorderMap`:
   - 构造文本边界二值图（threshold map）以及用于计算损失的 mask。

6. `NormalizeImage`:
   - 将图像元素值归一化到 `[-1, 1]` 的范围内。

7. `FilterKeys`:
   - 过滤掉不需要的键值对，包括 `polygons`、`filename`、`shape`、`ignore_tags` 和 `is_training`。

这些处理步骤将被按顺序应用于训练数据，以便在模型训练过程中对数据进行预处理和增强。

```python
train_processes = [
    BaseAugment(only_resize=False, keep_ratio=False,
        augmenters=iaa.Sequential([
            iaa.Fliplr(0.5),               # 水平翻转
            iaa.Affine(rotate=(-10, 10)),  # 旋转
            iaa.Resize((0.5, 3.0))         # 尺寸调整
        ])),
    ColorJitter(),                         # 颜色增强
    RandomCropData(size=[640, 640]),       # 随机裁剪
    MakeSegDetectionData(),                # 构造 probability map
    MakeBorderMap(),                       # 构造 threshold map
    NormalizeImage(),                      # 归一化
    FilterKeys(superfluous=['polygons', 'filename', 'shape', 'ignore_tags', 'is_training']),    # 过滤多余元素
]
```

### 数据集导入方法

这段代码定义了一个名为 `ImageDataset` 的数据集类，用于加载图像数据和相应的标注信息，并应用预处理步骤对数据进行处理。让我们逐步分析这段代码：

1. `__init__` 方法：
   - 初始化函数，用于设置数据集的路径、是否为训练模式以及数据处理步骤。
   - 接受参数 `data_dir` 和 `gt_dir`，分别表示图像数据和标注信息的路径。
   - `is_training` 表示是否为训练模式，`processes` 表示数据处理步骤列表。
   - 初始化图像路径列表 `self.image_paths` 和标注路径列表 `self.gt_paths`，并加载标注信息。

2. `load_ann` 方法：
   - 加载标注信息的辅助函数，用于从文本文件中读取标注信息并存储为列表形式。

3. `__getitem__` 方法：
   - 根据给定的索引加载图像数据和对应的标注信息。
   - 读取图像数据和标注信息，并将其存储在 `data` 字典中。
   - 根据是否提供了数据处理步骤，逐步应用这些步骤对数据进行处理。
   - 返回处理后的数据。

4. `__len__` 方法：
   - 返回数据集的长度，即图像路径列表的长度。

这个数据集类可以方便地加载图像数据和标注信息，并根据提供的处理步骤对数据进行处理。

```python
class ImageDataset(data.Dataset):
    def __init__(self, data_dir=None, gt_dir=None, is_training=True, processes=None):
        self.data_dir = data_dir
        self.gt_dir = gt_dir
        self.is_training = is_training
        self.processes = processes


        self.image_paths = []
        self.gt_paths = []


        image_list = os.listdir(self.data_dir)
        self.image_paths = [self.data_dir + '/' + t for t in image_list]
        self.gt_paths = [self.gt_dir + '/' + t.replace('.jpg', '.txt') for t in image_list]
        self.targets = self.load_ann()      # 导入标注信息


    def load_ann(self):
        res = []
        for gt in self.gt_paths:
            lines = []
            reader = open(gt, 'r').readlines()
            for line in reader:
                item = {}
                line = line.strip().split()
                poly = np.array(list(map(float, line[:8]))).reshape((-1, 2)).tolist()
                item['poly'] = poly
                item['text'] = line[8]   # 前8为 polygon 坐标，第9是文本字符串
                lines.append(item)
            res.append(lines)
        return res


    def __getitem__(self, index):
        if index >= len(self.image_paths):
            index = index % len(self.image_paths)
        data = {}
        image_path = self.image_paths[index]

        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
        target = self.targets[index]

        data['filename'] = image_path.split('/')[-1]
        data['data_id'] = image_path.split('/')[-1]
        data['image'] = img
        data['lines'] = target
        data['is_training'] = self.is_training
        if self.processes is not None:
            for data_process in self.processes:    # 做数据增强
                data = data_process.process(data)
        return data


    def __len__(self):
        return len(self.image_paths)
```

### 数据处理可视化

这段代码是用于数据处理可视化的部分，包括加载数据集、获取一个 batch，并对 batch 中的图像和相关数据进行可视化展示。让我们逐步分析这段代码：

创建 ImageDataset 实例 train_dataset：

使用 ImageDataset 类加载训练数据集。
设置数据目录、标注目录、训练模式和数据处理步骤。
创建数据加载器 train_dataloader：

使用 DataLoader 类创建数据加载器，用于加载 train_dataset 中的数据。
设置批量大小为 2，使用 0 个工作线程，打乱数据并不丢弃最后一批数据。
获取一个 batch 数据：

使用 next(iter(train_dataloader)) 获取一个 batch 数据。
画图：

使用 Matplotlib 绘制图像和相关数据的可视化结果。
展示图像、probability map、threshold map 和 threshold mask。
这段代码通过可视化展示了经过处理后的图像以及相应的概率图、阈值图和阈值 mask，有助于了解数据处理的效果和结果。

```python
# 数据处理可视化
train_dataset = ImageDataset(data_dir=det_args.train_dir, gt_dir=det_args.train_gt_dir, is_training=True, processes=train_processes)
train_dataloader = data.DataLoader(train_dataset, batch_size=2, num_workers=0, shuffle=True, drop_last=False)
batch = next(iter(train_dataloader))    # 获取一个 batch


# 画图
plt.figure(figsize=(60,60))
image = NormalizeImage.restore(batch['image'][0])
plt.subplot(141)
plt.title('image', fontdict={'size': 60})
plt.xticks([])
plt.yticks([])
plt.imshow(image)


probability_map = (batch['gt'][0].to('cpu').numpy() * 255).astype(np.uint8)
plt.subplot(142)
plt.title('probability_map', fontdict={'size': 60})
plt.xticks([])
plt.yticks([])
plt.imshow(probability_map, cmap='gray')


threshold_map = (batch['thresh_map'][0].to('cpu').numpy() * 255).astype(np.uint8)
plt.subplot(143)
plt.title('threshold_map', fontdict={'size': 60})
plt.xticks([])
plt.yticks([])
plt.imshow(threshold_map, cmap='gray')


threshold_mask = (batch['thresh_mask'][0].to('cpu').numpy() * 255).astype(np.uint8)
plt.subplot(144)
plt.title('threshold_mask', fontdict={'size': 60})
plt.xticks([])
plt.yticks([])
plt.imshow(threshold_mask, cmap='gray')
```

显示效果如下：

<div style="text-align:center;">
    <img src="https://github.com/LouisMao666/meter-identify/assets/149593046/73203406-e4e1-496e-8a76-254195a27473" alt="image">
</div>

### 计算 map 损失

这段代码定义了几种损失函数，用于训练文本检测模型。让我简要解释一下这些损失函数的作用：

1. **MaskL1Loss**: 这个损失函数用于计算 threshold map 的 L1 损失。在文本检测中，threshold map 通常表示文本的边界信息，这个损失函数的目的是使预测的 threshold map 与真实的 threshold map 之间的差异最小化。

2. **DiceLoss**: 这个损失函数用于计算 approximate binary map 的 Dice 损失。在文本检测中，approximate binary map 通常用于表示文本的二值分割结果，这个损失函数的目的是使预测的 approximate binary map 与真实的 binary map 之间的重叠最大化。

3. **L1BalanceCELoss**: 这个损失函数结合了 L1 损失、二分类交叉熵损失和 Dice 损失。通过权衡不同的损失函数，它可以更好地训练文本检测模型。其中，`l1_scale` 和 `bce_scale` 分别表示 L1 损失和二分类交叉熵损失的权重。

这些损失函数在训练过程中被用于计算模型预测值与真实标签之间的差异，并且被用作优化目标，以便通过反向传播算法来调整模型的参数，从而使模型的预测更加接近真实情况。

```python
# MaskL1Loss 计算 threshold map 损失
class MaskL1Loss(nn.Module):
    def __init__(self):
        super(MaskL1Loss, self).__init__()


    def forward(self, pred, gt, mask):
        mask_sum = mask.sum()
        if mask_sum.item() == 0:
            return mask_sum
        else:
            loss = (torch.abs(pred[:, 0] - gt) * mask).sum() / mask_sum
            return loss


# DiceLoss 计算 approximate binary map 损失
class DiceLoss(nn.Module):
    '''
    Loss function from https://arxiv.org/abs/1707.03237
    '''
    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps


    def forward(self, pred, gt, mask):
        if pred.dim() == 4:
            pred = pred[:, 0, :, :]
        assert pred.shape == gt.shape
        assert pred.shape == mask.shape


        interp = (pred * gt * mask).sum()
        union = (pred * mask).sum() + (gt * mask).sum() + self.eps
        loss = 1 - 2.0 * interp / union
        return loss


class L1BalanceCELoss(nn.Module):
    def __init__(self, eps=1e-6, l1_scale=10, bce_scale=1):
        super(L1BalanceCELoss, self).__init__()
        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss()
        self.bce_loss = BalanceCrossEntropyLoss()


        self.l1_scale = l1_scale        # 不同损失赋予不同权重
        self.bce_scale = bce_scale


    def forward(self, pred, batch):
        metrics = dict()
        bce_loss = self.bce_loss(pred['binary'], batch['gt'], batch['mask'])
        l1_loss = self.l1_loss(pred['thresh'], batch['thresh_map'], batch['thresh_mask'])
        dice_loss = self.dice_loss(pred['thresh_binary'], batch['gt'], batch['mask'])

        loss = dice_loss + self.l1_scale * l1_loss + bce_loss * self.bce_scale
        metrics['binary_loss'] = bce_loss
        metrics['thresh_loss'] = l1_loss
        metrics['thresh_binary_loss'] = dice_loss

        return loss, metrics
```

### 网络模型

这段代码定义了一个用于文本检测的神经网络模型。让我简要解释一下其结构和功能：

#### Backbone
- `Bottleneck`: 这是 ResNet 的基本模块。它由一系列卷积层和批归一化层组成，用于提取输入图像的特征。
- `ResNet`: 这是一个基于 ResNet 架构的主干网络。它包含多个不同层级的 Bottleneck 模块，用于逐步提取图像特征。ResNet 的最后输出是不同分辨率的特征图，供后续的分割网络使用。

#### Decoder
- `SegDetector`: 这是一个分割网络，用于从 ResNet 的特征图中生成文本检测的输出。它通过上采样和融合不同层级的特征图来生成最终的文本检测结果。

#### 损失函数
- `L1BalanceCELoss`: 这是一个综合了 L1 损失、二分类交叉熵损失和 Dice 损失的损失函数。它用于衡量模型预测值与真实标签之间的差异，帮助模型学习更好地进行文本检测。

#### 模型封装
- `BasicModel`: 这是将 Backbone 和 Decoder 组合在一起的基本模型。它负责将输入图像送入 Backbone 进行特征提取，然后将特征传递给 Decoder 进行文本检测。
- `SegDetectorModel`: 这是对 BasicModel 进行包装的模型。它负责将输入数据送入模型，并根据是否处于训练阶段来计算损失函数或生成预测结果。

通过这样的网络结构和损失函数，模型能够学习有效地进行文本检测任务。

```python
'''
Backbone
'''
class Bottleneck(nn.Module):
    expansion = 4


    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x


        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)


        out = self.conv3(out)
        out = self.bn3(out)


        if self.downsample is not None:
            residual = self.downsample(x)


        out += residual
        out = self.relu(out)


        return out


class ResNet(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.smooth = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=1)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )


        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))


        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)


        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)


        return x2, x3, x4, x5

'''
Decoder
'''
class SegDetector(nn.Module):
    def __init__(self, in_channels=[256, 512, 1024, 2048], inner_channels=256, k=50, bias=False):
        super(SegDetector, self).__init__()
        self.k = k
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')


        self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1, bias=bias)
        self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1, bias=bias)
        self.in3 = nn.Conv2d(in_channels[-3], inner_channels, 1, bias=bias)
        self.in2 = nn.Conv2d(in_channels[-4], inner_channels, 1, bias=bias)


        self.out5 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=8, mode='nearest'))
        self.out4 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=4, mode='nearest'))
        self.out3 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.out2 = nn.Conv2d(
            inner_channels, inner_channels//4, 3, padding=1, bias=bias)


        self.binarize = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels//4, inner_channels//4, 2, 2),
            nn.BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels//4, 1, 2, 2),
            nn.Sigmoid())
        self.binarize.apply(self.weights_init)


        self.thresh = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, inner_channels // 4, 2, 2),
            nn.BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, 1, 2, 2),
            nn.Sigmoid())
        self.thresh.apply(self.weights_init)


        self.in5.apply(self.weights_init)
        self.in4.apply(self.weights_init)
        self.in3.apply(self.weights_init)
        self.in2.apply(self.weights_init)
        self.out5.apply(self.weights_init)
        self.out4.apply(self.weights_init)
        self.out3.apply(self.weights_init)
        self.out2.apply(self.weights_init)


    def weights_init(self, m):
        # 模型权重初始化
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)


    def forward(self, features):
        c2, c3, c4, c5 = features
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)


        out4 = self.up5(in5) + in4   # 尺寸为输入图像的 1/16
        out3 = self.up4(out4) + in3  # 1/8
        out2 = self.up3(out3) + in2  # 1/4


        p5 = self.out5(in5)
        p4 = self.out4(out4)
        p3 = self.out3(out3)
        p2 = self.out2(out2)


        fuse = torch.cat((p5, p4, p3, p2), 1)    # 尺寸为 batch_size，64*4， H', W'
        binary = self.binarize(fuse)
        if self.training:
            result = OrderedDict(binary=binary)
            thresh = self.thresh(fuse)
            thresh_binary = self.step_function(binary, thresh)
            result.update(thresh=thresh, thresh_binary=thresh_binary)
            return result
        else:
            return binary    # for inference


    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))



# 包装
class BasicModel(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.backbone = ResNet()
        self.decoder = SegDetector()


    def forward(self, data):
        output = self.backbone(data)
        output = self.decoder(output)
        return output


class SegDetectorModel(nn.Module):
    def __init__(self, device):
        super(SegDetectorModel, self).__init__()
        self.model = BasicModel()
        self.criterion = L1BalanceCELoss()
        self.device = device
        self.to(self.device)


    def forward(self, batch, training=True):
        for key, value in batch.items():
            if value is not None and hasattr(value, 'to'):
                batch[key] = value.to(self.device)

        pred = self.model(batch['image'].float())


        if self.training:
            loss, metrics = self.criterion(pred, batch)    # 计算损失函数
            return pred, loss, metrics
        else:
            return pred
```

### 可视化模型训练过程中的数据和结果

这段代码用于可视化模型训练过程中的数据和结果，具体包括：

1. 获取一个训练批次的数据，并将其送入模型进行前向传播，计算损失函数和评价指标。
2. 打印出输入数据、模型 backbone 的输出、模型 decoder 的输出、以及损失函数的计算结果。
3. 绘制输入图像、ground truth 的 probability map、threshold map 以及模型预测的 approximate map。

这样的可视化过程有助于我们了解模型在训练过程中的表现情况，从而进行调试和优化。

如果你有任何问题或需要进一步的解释，请随时告诉我。

```python
train_dataset = ImageDataset(data_dir=det_args.train_dir, gt_dir=det_args.train_gt_dir, is_training=True, processes=train_processes)
train_dataloader = data.DataLoader(train_dataset, batch_size=2, num_workers=0, shuffle=True, drop_last=False)
batch = next(iter(train_dataloader))


# 模型各阶段数据格式
for key, value in batch.items():
    if value is not None and hasattr(value, 'to'):
        batch[key] = value.to(device)

model = SegDetectorModel(device)
model.train()
out1 = model.model.backbone(batch['image'].float())
out2 = model.model.decoder(out1)
loss, metrics = model.criterion(out2, batch)


# batch 输入
print('batch 输入：')
for key, value in batch.items():
    print(key, value.shape)
print()


# backbone 输出
print('backbone 输出：')
for o1 in out1:
    print(o1.shape)
print()


# decoder 输出
print('decoder 输出：')
for key, value in out2.items():
    print(key, value.shape)
print()


# criterion 输出
print('criterion 输出：')
print('loss: ', loss)
print(metrics)
print()


# 画图
plt.figure(figsize=(60,60))
image = NormalizeImage.restore(batch['image'][0])
plt.subplot(141)
plt.title('image', fontdict={'size': 60})
plt.xticks([])
plt.yticks([])
plt.imshow(image)


probability_map = (batch['gt'][0].to('cpu').numpy() * 255).astype(np.uint8)
plt.subplot(142)
plt.title('probability_map', fontdict={'size': 60})
plt.xticks([])
plt.yticks([])
plt.imshow(probability_map, cmap='gray')


threshold_map = (batch['thresh_map'][0].to('cpu').numpy() * 255).astype(np.uint8)
plt.subplot(143)
plt.title('threshold_map', fontdict={'size': 60})
plt.xticks([])
plt.yticks([])
plt.imshow(threshold_map, cmap='gray')


# 模型预测的 approximate map
binary = (out2['binary'][0][0].detach().to('cpu').numpy() * 255).astype(np.uint8)
plt.subplot(144)
plt.title('binary', fontdict={'size': 60})
plt.xticks([])
plt.yticks([])
plt.imshow(binary, cmap='gray')
```

#### 输出：

batch 输入：
image torch.Size([2, 3, 640, 640])
gt torch.Size([2, 640, 640])
mask torch.Size([2, 640, 640])
thresh_map torch.Size([2, 640, 640])
thresh_mask torch.Size([2, 640, 640])

backbone 输出：
torch.Size([2, 256, 160, 160])
torch.Size([2, 512, 80, 80])
torch.Size([2, 1024, 40, 40])
torch.Size([2, 2048, 20, 20])

decoder 输出：
binary torch.Size([2, 1, 640, 640])
thresh torch.Size([2, 1, 640, 640])
thresh_binary torch.Size([2, 1, 640, 640])

criterion 输出：
loss:  tensor(12.5495, device='cuda:0', grad_fn=<AddBackward0>)
{'binary_loss': tensor(7.8474, device='cuda:0', grad_fn=<DivBackward0>), 'thresh_loss': tensor(0.3727, device='cuda:0', grad_fn=<DivBackward0>), 'thresh_binary_loss': tensor(0.9749, device='cuda:0', grad_fn=<RsubBackward1>)}

![image](https://github.com/LouisMao666/meter-identify/assets/149593046/71b74762-024d-40b3-9ccd-024ca90cd6e7)

### 学习率调整方法

这段代码实现了一个学习率调整的方法 `DecayLearningRate`，并将其应用于检测模型的训练过程 `det_train()` 中。

在 `det_train()` 函数中，首先创建了模型、数据加载器和优化器。然后，通过 `DecayLearningRate` 类来调整学习率，根据当前的训练轮次和步数来更新学习率。在每个训练步骤中，计算损失、执行反向传播和优化器步骤。如果满足打印间隔条件，将打印出当前的训练信息，包括训练步数、轮次、损失、指标和学习率。最后，在每个保存间隔时保存模型的检查点，并在训练结束时保存最终模型。

这种学习率调整方法可以根据训练的进展情况自动地调整学习率，以提高模型的性能和收敛速度。

```python
# 学习率调整方法
class DecayLearningRate():
    def __init__(self, lr=0.004, epochs=200, factor=0.9):
        self.lr = lr
        self.epochs = epochs
        self.factor = factor


    def get_learning_rate(self, epoch):
        # 学习率随着训练过程进行不断下降
        rate = np.power(1.0 - epoch / float(self.epochs + 1), self.factor)
        return rate * self.lr

    def update_learning_rate(self, optimizer, epoch):
        lr = self.get_learning_rate(epoch)
        for group in optimizer.param_groups:
            group['lr'] = lr
        return lr




# 检测模型训练
def det_train():
    # model
    model = SegDetectorModel(device)


    # data_loader
    train_dataset = ImageDataset(data_dir=det_args.train_dir, gt_dir=det_args.train_gt_dir, is_training=True, processes=train_processes)
    train_dataloader = data.DataLoader(train_dataset, batch_size=det_args.batch_size, num_workers=det_args.num_workers, shuffle=True, drop_last=False)

    # initialize dataloader and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=det_args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = DecayLearningRate(lr=det_args.lr, epochs=det_args.max_epoch)


    step = 0
    epoch = 0
    model.train()
    os.makedirs(det_args.save_dir, exist_ok=True)
    for epoch in range(det_args.max_epoch):
        for batch in train_dataloader:
            step += 1
            current_lr = scheduler.update_learning_rate(optimizer, epoch)    # 学习率调整


            optimizer.zero_grad()
            pred, loss, metrics = model.forward(batch, training=True)
            loss.backward()
            optimizer.step()


            if step % det_args.print_interval == 0:
                line = []
                for key, l_val in metrics.items():
                    line.append('{0}: {1:.4f}'.format(key, l_val.mean()))
                line = '\t'.join(line)
                info = '\t'.join(['step:{:6d}', 'epoch:{:3d}', 'loss: {:4f}', '{}', 'lr: {:.4f}']).format(step, epoch, loss.item(), line, current_lr)
                print(info)
                #if DEBUG:   # DEBUG 这里只跑少量数据
                    #break

        # 保存阶段模型
        if epoch % det_args.save_interval == 0:
            save_name = 'checkpoint_' + str(epoch)
            torch.save(model.state_dict(), os.path.join(det_args.save_dir, save_name))
            torch.save(model.state_dict(), det_args.saved_model_path)
        #if DEBUG:
            #break


    # 保存最终模型
    torch.save(model.state_dict(), det_args.saved_model_path)
```

### 由于电脑性能，只训练了10个epoch，训练过程的输出如下：

```css
<div style="background-color:black; color:white">
batch 输入：<br>
image torch.Size([2, 3, 640, 640])<br>
gt torch.Size([2, 640, 640])<br>
mask torch.Size([2, 640, 640])<br>
thresh_map torch.Size([2, 640, 640])<br>
thresh_mask torch.Size([2, 640, 640])<br>

backbone 输出：<br>
torch.Size([2, 256, 160, 160])<br>
torch.Size([2, 512, 80, 80])<br>
torch.Size([2, 1024, 40, 40])<br>
torch.Size([2, 2048, 20, 20])<br>

decoder 输出：<br>
binary torch.Size([2, 1, 640, 640])<br>
thresh torch.Size([2, 1, 640, 640])<br>
thresh_binary torch.Size([2, 1, 640, 640])<br>

criterion 输出：<br>
loss: tensor(12.5495, device='cuda:0', grad_fn=<AddBackward0>)<br>
{'binary_loss': tensor(7.8474, device='cuda:0', grad_fn=<DivBackward0>), 'thresh_loss': tensor(0.3727, device='cuda:0', grad_fn=<DivBackward0>), 'thresh_binary_loss': tensor(0.9749, device='cuda:0', grad_fn=<RsubBackward1>)}
</div>
```

### 文本检测的后处理方法

这段代码是一个用于文本检测的后处理方法，从预测的分割图中提取文本框的过程。以下是对该代码的主要步骤的解释：

1. **`represent` 方法**：接受一个批次的图像和对应的预测结果，并对每张图像的预测结果进行处理，得到文本框的坐标和相应的得分。

2. **`polygons_from_bitmap` 方法**：该方法从预测的分割图中提取文本框的多边形轮廓。它首先找到二值化后的分割图中的轮廓，然后对每个轮廓进行处理。对于每个轮廓，它进行多边形拟合以得到更加紧凑的表示，并计算一个得分以过滤掉低质量的候选框。在确定最终的多边形时，它对拟合得到的多边形进行了一定程度的扩张，并计算了最小外接矩形。

3. **`unclip` 方法**：该方法对文本框的多边形进行一定程度的扩张，以覆盖更多的文本区域。

4. **`get_mini_boxes` 方法**：该方法用于计算文本框的最小外接矩形。

5. **`box_score_fast` 方法**：该方法用于计算文本框区域的得分，以确定其是否包含文本。它计算了文本框内像素点的平均得分作为最终得分。

总体而言，这段代码利用了分割图和预测概率图来提取文本框。它采用多边形拟合和一些后处理步骤来获取更加准确的文本框表示。

```python
# 从分割图得到最终文字坐标的后处理方法
class SegDetectorRepresenter():
    '''
    从 probability map 得到检测框的方法
    '''
    def __init__(self, thresh=0.3, box_thresh=0.7, max_candidates=100):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.min_size = 3
        self.scale_ratio = 0.4


    def represent(self, batch, pred):
        images = batch['image']
        segmentation = pred > self.thresh    # 将预测分割图进行二值化
        boxes_batch = []
        scores_batch = []
        for batch_index in range(images.size(0)):
            height, width = batch['shape'][batch_index]
            boxes, scores = self.polygons_from_bitmap(pred[batch_index], segmentation[batch_index], width, height)
            boxes_batch.append(boxes)
            scores_batch.append(scores)
        return boxes_batch, scores_batch



    def polygons_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        assert _bitmap.size(0) == 1
        bitmap = _bitmap.cpu().numpy()[0]
        pred = pred.cpu().detach().numpy()[0]
        height, width = bitmap.shape
        boxes = []
        scores = []


        contours, _ = cv2.findContours((bitmap*255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # 找分割轮廓


        for contour in contours[:self.max_candidates]:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)      # 多边形拟合轮廓曲线
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            score = self.box_score_fast(pred, points.reshape(-1, 2))   # 计算分割区域的整体得分，去除低分候选区域
            if self.box_thresh > score:
                continue

            if points.shape[0] > 2:
                box = self.unclip(points, unclip_ratio=2.0)    # 因为得到的分割结果是文本收缩区域，因此需要进行一定程度扩张
                if len(box) != 1:
                    continue
            else:
                continue

            box = box.reshape(-1, 2)
            mini_box, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))   # 计算最小外接矩形
            if sside < self.min_size + 2:
                continue


            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            mini_box[:, 0] = np.clip(np.round(mini_box[:, 0] / width * dest_width), 0, dest_width)       # 尺寸与原图对齐
            mini_box[:, 1] = np.clip(np.round(mini_box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(mini_box.tolist())
            scores.append(score)
        return boxes, scores
```

### 测试图片处理

这段代码包含两个图像处理函数：

1. **`resize_image` 函数**：这个函数对图像进行缩放，使其最短边达到预设的长度。如果图像的高度小于宽度，则将高度调整为预设长度，然后根据原始图像的尺寸比例缩放宽度；反之亦然。最后，确保缩放后的宽度和高度都是32的倍数，这通常有助于某些模型的运算效率。

2. **`load_test_image` 函数**：这个函数用于加载测试图像并进行预处理。它首先使用OpenCV加载图像，然后调用`resize_image`函数对图像进行缩放。接下来，将RGB均值减去每个像素的RGB均值，然后将像素值除以255进行归一化。最后，将图像转换为PyTorch张量，并调整通道顺序以适应模型输入的格式。

这些函数组合起来，能够加载和预处理测试图像，使其适用于模型输入。

```python
# 测试图片处理
def resize_image(img):
    # 图像最短边设定为预设长度，长边根据原图尺寸比例进行缩放
    height, width, _ = img.shape
    if height < width:
        new_height = det_args.test_img_short_side
        new_width = int(math.ceil(new_height / height * width / 32) * 32)
    else:
        new_width = det_args.test_img_short_side
        new_height = int(math.ceil(new_width / width * height / 32) * 32)
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img

def load_test_image(image_path):
    RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
    img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
    original_shape = img.shape[:2]
    img = resize_image(img)
    img -= RGB_MEAN
    img /= 255.
    img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
    return img, original_shape
```

### 检测结果输出

这段代码负责将检测结果格式化输出到文本文件中。

- `det_res_dir` 是输出结果的目录。
- `batch` 包含了一个批次的图像信息，其中可能包括图像的文件名、尺寸等。
- `output` 是模型的输出结果，包括检测到的文本框坐标和对应的置信度得分。

函数首先遍历每张图像，获取原始图像的形状和文件名等信息。然后，根据文件名生成输出结果的文件路径。接着，对于每个检测到的文本框，将其坐标和对应的置信度得分写入到结果文件中。最终，每个文本框的信息都以逗号分隔，并写入一行。

这样，每张输入图像都会对应一个输出的文本文件，文件中包含了检测到的文本框的坐标和置信度得分。

```python
# 检测结果输出
def format_output(det_res_dir, batch, output):
    batch_boxes, batch_scores = output
    for index in range(batch['image'].size(0)):
        original_shape = batch['shape'][index]
        filename = batch['filename'][index]
        result_file_name = 'det_res_' + filename.split('/')[-1].split('.')[0] + '.txt'
        result_file_path = os.path.join(det_res_dir, result_file_name)
        boxes = batch_boxes[index]
        scores = batch_scores[index]
        with open(result_file_path, 'wt') as res:
            for i, box in enumerate(boxes):
                box = np.array(box).reshape(-1).tolist()
                result = ",".join([str(int(x)) for x in box])
                score = scores[i]
                res.write(result + ',' + str(score) + "\n")
```

### 测试

#### 函数定义

这段代码是用于进行读数区域检测的测试的函数。下面是函数的主要步骤：

1. **模型加载**：加载之前训练的读数区域检测模型。

2. **后处理**：创建一个 `SegDetectorRepresenter` 的实例，用于对模型输出进行后处理，得到最终的文本框坐标和置信度得分。

3. **推理**：遍历测试图像目录中的每张图像，对每张图像进行推理。首先加载图像并进行预处理，然后将其传递给模型进行推理，得到模型输出。接着，使用后处理方法获取最终的文本框结果，并将结果格式化输出到文本文件中。

4. **结果保存**：将输出的文本结果保存到指定的目录中。

这个函数的作用是对测试集中的图像进行文本检测，并将检测结果保存到文件中。

```python
def det_test():
    # 模型加载
    model = SegDetectorModel(device)
    model.load_state_dict(torch.load(det_args.saved_model_path, map_location=device), strict=False)
    model.eval()

    # 后处理
    representer = SegDetectorRepresenter(thresh=det_args.thresh, box_thresh=det_args.box_thresh, max_candidates=det_args.max_candidates)

    # 推理
    os.makedirs(det_args.det_res_dir, exist_ok=True)
    batch = dict()
    cnt = 0
    with torch.no_grad():
        for file in tqdm(os.listdir(det_args.test_dir)):
            img_path = os.path.join(det_args.test_dir, file)
            image, ori_shape = load_test_image(img_path)
            batch['image'] = image
            batch['shape'] = [ori_shape]
            batch['filename'] = [file]
            pred = model.forward(batch, training=False)
            output = representer.represent(batch, pred)
            format_output(det_args.det_res_dir, batch, output)


            #if DEBUG and cnt >= 6:    # DEBUG
                #break
            cnt += 1
```

#### 测试结果可视化

这段代码用于将文本检测的结果可视化，以便直观地查看模型的检测效果。主要步骤如下：

1. **创建可视化结果保存目录**：指定一个目录用于保存可视化结果。

2. **遍历检测结果文件**：遍历文本检测结果目录中的每个结果文件。

3. **加载图像**：根据检测结果文件的文件名加载相应的原始图像。

4. **读取检测结果**：从对应的检测结果文件中读取文本框的坐标信息。

5. **绘制文本框**：将检测到的文本框绘制在原始图像上。

6. **保存结果图像**：将带有文本框的图像保存到可视化结果目录中。

7. **可视化显示**：选择前5张图像进行显示，并将它们显示在一个大的画布中，以便观察检测效果。

这样，可以通过可视化结果来验证文本检测模型的性能，以及检查其在不同图像上的表现。

```python
# 检测结果可视化
test_dir = 'data/test_imgs'
det_dir = 'temp/det_res'
det_vis_dir = 'temp/det_vis_test'


os.makedirs(det_vis_dir, exist_ok=True)
label_files = os.listdir(det_dir)
cnt = 0
plt.figure(figsize=(60,60))
for label_file in tqdm(label_files):
    if not label_file.endswith('.txt'):
        continue
    image = cv2.imread(os.path.join(test_dir, label_file.replace('det_res_', '')[:-4] + '.jpg'))

    with open(os.path.join(det_dir, label_file), 'r') as f:
        lines = f.readlines()

    save_name = label_file.replace('det_res_', '')[:-4]+'.jpg'
    if len(lines) == 0:
        cv2.imwrite(os.path.join(det_vis_dir, save_name), image)
    else:
        line = lines[0].strip().split(',')
        locs = [float(t) for t in line[:8]]


        # draw box
        locs = np.array(locs).reshape(1, -1, 2).astype(np.int32)
        image = cv2.imread(os.path.join(test_dir, label_file.replace('det_res_', '')[:-4] + '.jpg'))
        cv2.polylines(image, locs, True, (255, 255, 0), 8)

        # save images
        save_name = label_file.replace('det_res_', '')[:-4]+'.jpg'
        cv2.imwrite(os.path.join(det_vis_dir, save_name), image)

    if cnt < 5:    # 只画5张
        plt.subplot(151 + cnt)
        plt.title(save_name, fontdict={'size': 60})
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image)
        cnt += 1
```

#### 显示效果

由于训练批次较少，第一张和第五张图片的识别效果欠佳

![image](https://github.com/LouisMao666/meter-identify/assets/149593046/ff14f168-5f10-441e-849d-3f70a182fef3)

## 第二部分：水表读数识别

这部分我们处理的对象，已经从整张图片变成了切割完成的读数区域，那么问题也就变得简单，相信大家初学时都练习过MNIST数据集，二者本质上非常接近。

### 参数设定

这个类定义了用于文本识别任务的参数选项，以下是每个参数的含义和默认值：

1. `height`：图像的高度，用于输入文本识别模型的图像高度，默认值为32。

2. `width`：图像的宽度，用于输入文本识别模型的图像宽度，默认值为100。

3. `voc_size`：字符的数量，包括文本中可能出现的字符以及额外的用于填充的字符（PADDING位），默认值为21。

4. `decoder_sdim`：解码器（decoder）的隐藏状态的维度，默认值为512。

5. `max_len`：文本的最大长度，默认值为5。

6. `lr`：初始学习率，默认值为1.0。

7. `milestones`：在训练过程中降低学习率的里程碑（epoch数），默认值为[40, 60]，即在第40和第60个epoch时降低学习率。

8. `max_epoch`：训练的最大epoch数，默认值为80。

9. `batch_size`：训练时的批次大小，默认值为64。

10. `num_workers`：用于数据加载的线程数，默认值为0。

11. `print_interval`：打印训练信息的间隔（单位：batch），默认值为25。

12. `save_interval`：保存模型的间隔（单位：batch），默认值为125。

13. `train_dir`：训练数据集的目录路径，默认值为'temp/rec_datasets/train_imgs'。

14. `test_dir`：测试数据集的目录路径，默认值为'temp/rec_datasets/test_imgs'。

15. `save_dir`：保存训练好的模型的目录路径，默认值为'temp/rec_models/'。

16. `saved_model_path`：加载已保存的模型的路径，默认值为'temp/rec_models/checkpoint_final'。

17. `rec_res_dir`：文本识别结果的保存目录，默认值为'temp/rec_res/'。

这些参数提供了在进行文本识别任务时可能需要调整的各种选项，并且每个参数都有默认值，因此在大多数情况下可以直接使用默认设置进行训练或测试。

```python
class RecOptions():
    def __init__(self):
        self.height = 32              # 图像尺寸
        self.width = 100
        self.voc_size = 21            # 字符数量 '0123456789ABCDEFGHIJ' + 'PADDING'位
        self.decoder_sdim = 512
        self.max_len = 5              # 文本长度
        self.lr = 1.0
        self.milestones = [40, 60]    # 在第 40 和 60 个 epoch 训练时降低学习率
        self.max_epoch = 80
        self.batch_size = 64
        self.num_workers = 0
        self.print_interval = 25
        self.save_interval = 125
        self.train_dir = 'temp/rec_datasets/train_imgs'
        self.test_dir = 'temp/rec_datasets/test_imgs'
        self.save_dir = 'temp/rec_models/'
        self.saved_model_path = 'temp/rec_models/checkpoint_final'
        self.rec_res_dir = 'temp/rec_res/'


    def set_(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)


rec_args = RecOptions()
```

### 数据预处理

文本识别训练数据的预处理，包括标签处理和构造识别训练数据。

1. **标签处理**：根据预定义的半字符类别字典`EXT_CHARS`，对多个字符的情况进行处理。如果标签中包含多个字符，算法会检查每个字符的相同位置是否相同。如果相同位置的字符相同，则将其保留；如果相同位置的字符不同，则根据`EXT_CHARS`字典将其映射到半字符类别中。处理完成后得到一个新的标签`ext_word`。

2. **图像调整**：根据文本的倾斜角度，将倾斜的文字图像调整为水平图像。这里使用透视变换`cv2.warpPerspective`来实现调整。首先计算倾斜文字图像的宽度`w`和高度`h`，然后根据透视变换矩阵`M`将倾斜的文字图像调整为水平图像。

3. **保存处理后的图像**：将处理后的文本图像保存到指定的目录中，文件名由原始文件名、半字符类别和`.jpg`后缀组成。

这样，通过预处理代码，可以构造出适合文本识别模型训练的数据集。

```python
标签处理：定义新字符类处理半字符的情况，比如将'0-1半字符'归到'A'类，减小歧义
识别训练数据构造：从完整图像中裁剪出文本图像作为识别模型输入数据
'''
def PreProcess():
    EXT_CHARS = {
        '01': 'A', '12': 'B', '23': 'C', '34': 'D', '45': 'E',
        '56': 'F', '67': 'G', '78': 'H', '89': 'I', '09': 'J'
    }


    train_dir = 'data/train_imgs'
    train_labels_dir = 'data/labels'
    word_save_dir = 'temp/rec_datasets/train_imgs'      # 保存识别训练数据集
    os.makedirs(word_save_dir, exist_ok=True)
    label_files = os.listdir(train_labels_dir)
    for label_file in tqdm(label_files):
        with open(os.path.join(train_labels_dir, label_file), 'r') as f:
            lines = f.readlines()
        line = lines[0].strip().split()
        locs = line[:8]
        words = line[8:]

        # 标签处理
        if len(words) == 1:
            ext_word = words[0]
        else:
            assert len(words) % 2 == 0
            ext_word = ''
            for i in range(len(words[0])):
                char_i = [word[i] for word in words]
                if len(set(char_i)) == 1:
                    ext_word += char_i[0]
                elif len(set(char_i)) == 2:
                    char_i = list(set(char_i))
                    char_i.sort()
                    char_i = ''.join(char_i)
                    ext_char_i = EXT_CHARS[char_i]
                    ext_word += ext_char_i


        locs = [int(t) for t in line[:8]]

        # 将倾斜文字图像调整为水平图像
        x1, y1, x2, y2, x3, y3, x4, y4 = locs
        w = int(0.5 * (((x2-x1)**2+(y2-y1)**2)**0.5 + ((x4-x3)**2+(y4-y3)**2)**0.5))
        h = int(0.5 * (((x2-x3)**2+(y2-y3)**2)**0.5 + ((x4-x1)**2+(y4-y1)**2)**0.5))
        src_points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype='float32')
        dst_points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype='float32')
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        image = cv2.imread(os.path.join(train_dir, label_file.replace('.txt', '.jpg')))
        word_image = cv2.warpPerspective(image, M, (w, h))

        # save images
        cv2.imwrite(os.path.join(word_save_dir, label_file.replace('.txt', '')+'_'+ext_word+'.jpg'), word_image)


# 运行识别训练数据前处理代码
PreProcess()
```

### 数据集导入方法

这个类用于构建一个数据集，用于训练文本识别模型。以下是这个类的主要功能和属性：

1. **`__init__` 方法**：初始化函数，用于设置数据集的参数和加载数据。参数包括数据集目录 `data_dir`、文本最大长度 `max_len`、图像尺寸调整参数 `resize_shape` 和是否用于训练 `train`。在初始化过程中，会遍历数据集目录，获取所有图像文件的路径和对应的文本标签，并生成标签映射字典。

2. **`__len__` 方法**：返回数据集的大小，即图像的数量。

3. **`gen_labelmap` 方法**：静态方法，用于生成字符和数字标签的对应字典。其中，将字符映射到数字标签，同时将填充字符（PADDING）映射到0。

4. **`__getitem__` 方法**：获取数据集中指定索引的图像和标签。首先根据索引获取图像路径和对应的文本标签，然后根据文本标签构建标签向量。如果数据用于训练且存在数据增强操作，则对图像进行数据增强。最后将图像、标签向量和标签长度作为输出返回。

这个类提供了一个数据集对象，可以用于加载训练或测试数据，并根据需要进行数据增强操作。

```python
# data
class WMRDataset(data.Dataset):
    def __init__(self, data_dir=None, max_len=5, resize_shape=(32, 100), train=True):
        super(WMRDataset, self).__init__()
        self.data_dir = data_dir
        self.max_len = max_len
        self.is_train = train

        self.targets = [[os.path.join(data_dir, t), t.split('_')[-1][:5]] for t in os.listdir(data_dir) if t.endswith('.jpg')]
        self.PADDING, self.char2id, self.id2char = self.gen_labelmap()

        # 数据增强
        self.transform = transforms.Compose([
            transforms.Resize(resize_shape),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            # 可以添加更多的数据增强操作，比如 gaussian blur、shear 等
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.targets)

    @staticmethod
    def gen_labelmap(charset='0123456789ABCDEFGHIJ'):
        # 构造字符和数字标签对应字典
        PADDING = 'PADDING'
        char2id = {t:idx for t, idx in zip(charset, range(1, 1+len(charset)))}
        char2id.update({PADDING:0})
        id2char = {v:k for k, v in char2id.items()}
        return PADDING, char2id, id2char



    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_path = self.targets[index][0]
        word = self.targets[index][1]
        img = Image.open(img_path)

        label = np.full((self.max_len,), self.char2id[self.PADDING], dtype=np.int0)
        label_list = []
        word = word[:self.max_len]
        for char in word:
            label_list.append(self.char2id[char])

        label_len = len(label_list)
        assert len(label_list) <= self.max_len
        label[:len(label_list)] = np.array(label_list)

        if self.transform is not None and self.is_train:
            img = self.transform(img)
            img.sub_(0.5).div_(0.5)

        label_len = np.array(label_len).astype(np.int32)
        label = np.array(label).astype(np.int32)

        return img, label, label_len        # 输出图像、文本标签、标签长度, 计算 CTC loss 需要后两者信息
```

调用这个类

这段代码是使用定义的 WMRDataset 类构建数据集，并使用 PyTorch 的 DataLoader 创建数据加载器，然后获取一个批次的数据，并对图像进行可视化。

具体步骤如下：

构建数据集对象：使用 WMRDataset 类构建数据集对象 dataset，并传入训练数据集的目录、文本最大长度、图像尺寸调整参数和是否用于训练的标志。

创建数据加载器：使用 DataLoader 类创建数据加载器 train_dataloader，并传入数据集对象、批次大小、是否对数据进行洗牌、是否将数据加载到 GPU 的钉住内存中以加速训练、是否丢弃最后一个不完整的批次。

获取一个批次的数据：使用 iter(train_dataloader) 获取数据加载器的迭代器，并调用 next 方法获取一个批次的数据，包括图像、标签和标签长度。

图像可视化：将获取的图像数据进行可视化，并添加标题“image”。

标签解码：将标签从数字表示转换为字符串表示，并打印出来。

请注意，在将图像数据从张量转换为 NumPy 数组时，需要进行适当的缩放和格式转换，以便将其显示为图像。同时，将标签从数字表示转换为字符串表示时，需要使用数据集对象中定义的映射关系。

```python
dataset = WMRDataset(rec_args.train_dir, max_len=5, resize_shape=(rec_args.height, rec_args.width), train=True)
train_dataloader = data.DataLoader(dataset, batch_size=2, shuffle=True, pin_memory=True, drop_last=False)
batch = next(iter(train_dataloader))


image, label, label_len = batch
image = ((image[0].permute(1, 2, 0).to('cpu').numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
plt.title('image')
plt.xticks([])
plt.yticks([])
plt.imshow(image)

label_digit = label[0].to('cpu').numpy().tolist()
label_str = ''.join([dataset.id2char[t] for t in label_digit if t > 0])


print('label_digit: ', label_digit)
print('label_str: ', label_str)
```

#### 输出：

为了提高训练效率 这里使用的坐标并不是上一部分计算得出的，而是数据集中自带的正确坐标，将训练分为两个部分进行可以显著调高效率

读数区域的坐标 读数 显示读数区域

![2024-02-25](https://github.com/LouisMao666/meter-identify/assets/149593046/4846dca6-d8aa-4ab9-98c3-c76c3e5b178d)






























