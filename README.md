# 基于YOLOv7的单目3D检测方案

## 一、数据集
**百度rope3d，路端数据**

代码工作：
1. 数据可视化检查，修正错误标注（根据比赛冠军方案介绍，rope3d数据中存在Z轴方向错误）
2. 数据加载及转换：加载rope3d标注，根据相机参数，将目标3d中心（x,y,z）转换为图像投影中心点（xc, yc)，模型预测输出 [2dbbox，3d投影中心点（xc,yc）, 目标深度z，目标尺寸（w，h，l），目标角度yaw, 置信度，类别得分]
3. 数据增强：3d随机缩放，3d随机裁剪，随机左右翻转，均需要修改相机参数及label。

## 二、模型架构
先采用基础版的YOLOv7，后续可升级YOLOv7 E6E

代码工作：
1. 网络主干下采样至128倍（增加2组模块）， 增加head至5个(P6, P7)；
2. 修改detectHead部分代码，增加模型预测输出的维度；
3. 增加3d信息后处理，基于模型输出结果，在2d nms（修改输出尺度）后，对剩余目标计算3d位置信息，根据3d投影中心点（xc,yc）、 目标深度z和相机参数矩阵。再跟预测的目标尺寸、角度进行结果拼接，形成最终输出

## 三、损失函数
除原有的2d损失（cls, obj, bbox）之外，增加3dbbox损失

代码工作：
1. build_targets 

    根据loss选择，进行标签转换

2. 3d bbox loss 
    
    a. **使用3d bbox的8个角点做L1 loss(smoke 方案)** 

    使用预测的3d投影中心点（xc,yc）, 目标深度z，目标尺寸（w，h，l），目标角度yaw，计算3d包围框的8个角点，与标签值做L1 loss

    b. **使用多个loss之和（多种3d检测方法常用）**

    L1（3d中心点投影）+ L1（深度z）+ 角度损失

    角度损失常用的有两种方式：直接回归观察角L1 alpha（[cos(alpha), sin(alpha)]）代替ry，或者mutil bin loss （设置n个bin，每个bin覆盖2PI/n 个角度，预测bin的n分类和单个bin的offset）

    bin loss

    ```
    # 角度 head设置，获取主干网络的结果feature，回归两个结果dir_cls和 dir_reg
    dir_feat = nn.Sequential(
            nn.Conv2d(in_ch, feat_ch, kernel_size=3, padding=1),
            AttnBatchNorm2d(feat_ch, 10, momentum=0.03, eps=0.001),
            nn.ReLU(inplace=True))
    dir_cls = nn.Sequential(nn.Conv2d(feat_ch, self.num_alpha_bins, kernel_size=1))
    dir_reg = nn.Sequential(nn.Conv2d(feat_ch, self.num_alpha_bins, kernel_size=1))

    feat = backbone(input)
    alpha_feat = dir_feat(feat)
    alpha_cls_pred = dir_cls(alpha_feat)
    alpha_offset_pred = dir_reg(alpha_feat)

    pred_dict = {
        'alpha_cls_pred': alpha_cls_pred,
        'alpha_offset_pred': alpha_offset_pred
    }

    def decode_alpha(self, 
                    alpha_cls: torch.Tensor, 
                    alpha_offset: torch.Tensor) -> torch.Tensor:
        
        # Bin Class and Offset
        _, cls = alpha_cls.max(dim=-1)
        cls = cls.unsqueeze(2)
        alpha_offset = alpha_offset.gather(2, cls)
        
        # Convert to Angle
        angle_per_class = (2 * PI) / float(self.num_alpha_bins)
        angle_center = (cls * angle_per_class)
        alpha = (angle_center + alpha_offset)

        # Refine Angle
        alpha[alpha > PI] = alpha[alpha > PI] - (2 * PI)
        alpha[alpha < -PI] = alpha[alpha < -PI] + (2 * PI)
        return alpha

    # 损失设置：cls 交叉熵损失loss_alpha_cls，reg L1损失loss_alpha_reg

    crit_alpha_cls = CrossEntropyLoss(use_sigmoid=True, loss_weight=1.0)
    crit_alpha_reg = L1Loss(loss_weight=1.0)

    # Bin Classification
    alpha_cls_pred = extract_input(pred_dict['alpha_cls_pred'], indices, mask_target)
    alpha_cls_target = extract_target(target_dict['alpha_cls_target'], mask_target).type(torch.LongTensor)
    alpha_cls_onehot_target = alpha_cls_target\
        .new_zeros([len(alpha_cls_target), self.num_alpha_bins])\
        .scatter_(1, alpha_cls_target.view(-1, 1), 1).to(device)
    
    if mask_target.sum() > 0:
        loss_alpha_cls = crit_alpha_cls(alpha_cls_pred, alpha_cls_onehot_target)
    else:
        loss_alpha_cls = 0.0
    
    # Bin Offset Regression
    alpha_offset_pred = extract_input(pred_dict['alpha_offset_pred'], indices, mask_target)
    alpha_offset_pred = torch.sum(alpha_offset_pred * alpha_cls_onehot_target, 1, keepdim=True)
    alpha_offset_target = extract_target(target_dict['alpha_offset_target'], mask_target)
    
    loss_alpha_reg = crit_alpha_reg(alpha_offset_pred, alpha_offset_target)
    ```

3. 损失权重，需进行实验设置

## 四、训练过程
按照常规端到端训练方法进行训练和调参

代码工作：
1. 每个epoch后的评测需增加验证集3d AP的计算，并log结果和绘制图像；同时记录增加的loss结果和图像绘制
2. 每个epoch随机可视化一部分验证集3d bbox效果
3. 使用官方评测工具（仅有4个相似度结果计算，无3d AP和rope score代码），结合上述3d AP，增加官方Ropescore结果（基于IOU阈值0.5）

## 五、主要代码工作量
1. 数据可视化及错误修正：可借用官方可视化工具，增加代码查找错误标注图片，并形成可视化结果
2. 数据增强：可部分借用MonoCon代码
3. Loss计算
4. 3d信息后处理
5. 3d Ap计算，可部分借用kitti代码