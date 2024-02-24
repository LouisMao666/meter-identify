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

<div style="text-align:center;">
    <img src="https://github.com/LouisMao666/meter-identify/assets/149593046/73203406-e4e1-496e-8a76-254195a27473" alt="image">
</div>

### 



















