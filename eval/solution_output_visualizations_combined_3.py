import mmcv
import torch
from mmdet.apis import init_detector, inference_detector # inference_detector 在 MMDetection 3.x 中返回 DetDataSample
from mmdet.utils import register_all_modules
from mmdet.visualization import DetLocalVisualizer # 导入 DetLocalVisualizer
import os
import os.path as osp
import numpy as np # DetLocalVisualizer 需要图像为 numpy 数组

def visualize_model_predictions():
    # --- Configuration for Models ---
    models_to_visualize = [
        {
            'name': 'Mask_R-CNN',
            'config_file': '/data/jiangzishang/lty/作业/张力/mmdetection/configs/mask_rcnn/mask-rcnn_r50_fpn_1x_voc.py',
            'checkpoint_file': '/data/jiangzishang/lty/作业/张力/mmdetection/work_dirs/mask_rcnn_voc/epoch_50.pth'
        },
        {
            'name': 'Sparse_R-CNN',
            'config_file': '/data/jiangzishang/lty/作业/张力/mmdetection/configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_voc.py',
            'checkpoint_file': '/data/jiangzishang/lty/作业/张力/mmdetection/work_dirs/sparse_rcnn_voc_cocoYU/epoch_25.pth'
        }
    ]

    # !!! 重要: 请将以下列表替换为您想要可视化的图像的实际完整路径 !!!
    image_paths = [
        '/data/jiangzishang/lty/作业/张力/mmdetection/data/coco/VOC2007/JPEGImages/009952.jpg',
        '/data/jiangzishang/lty/作业/张力/mmdetection/data/coco/VOC2007/JPEGImages/009953.jpg',
        '/data/jiangzishang/lty/作业/张力/mmdetection/data/coco/VOC2007/JPEGImages/009954.jpg',
        # 添加更多图片路径...
    ]

    output_dir_base = './output_visualizations_combined_3/'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    score_threshold = 0.3 # 可视化时使用的分数阈值

    # --- End Configuration ---

    if not image_paths:
        print("错误: `image_paths` 列表为空。请提供图像路径。")
        return

    # 注册 MMDetection 模块
    try:
        # 对于 MMDetection 3.x, init_default_scope 需要设置为项目根目录对应的名称空间，
        # 或者如果您的自定义模块已经正确注册，直接调用即可。
        # 如果不确定，可以尝试两种方式。
        # 首先尝试 MMDetection 3.x 中常见的做法，如果 mmdet 是你的顶层包
        register_all_modules(init_default_scope='mmdet')
        print("MMDetection模块已使用 init_default_scope='mmdet' (MMDet 3.x 风格) 注册。")
    except Exception as e1:
        print(f"使用 init_default_scope='mmdet' 注册失败: {e1}")
        try:
            register_all_modules(init_default_scope=True)
            print("MMDetection模块已使用 init_default_scope=True (另一种 MMDet 3.x 风格) 注册。")
        except Exception as e2:
            print(f"使用 init_default_scope=True 注册失败: {e2}")
            try:
                register_all_modules(init_default_scope=False) # MMDetection 2.x 风格
                print("MMDetection模块已使用 init_default_scope=False (MMDet 2.x 风格) 注册。")
            except Exception as e_reg:
                print(f"所有注册MMDetection模块的尝试均失败: {e_reg}")
                return

    for model_info in models_to_visualize:
        model_name = model_info['name']
        config_file = model_info['config_file']
        checkpoint_file = model_info['checkpoint_file']

        print(f"\n--- 正在处理模型: {model_name} ---")
        output_dir_model = osp.join(output_dir_base, model_name)
        os.makedirs(output_dir_model, exist_ok=True)
        print(f"可视化结果将保存在: {osp.abspath(output_dir_model)}")

        # 加载模型
        print(f"正在加载 {model_name} 从: {checkpoint_file}")
        model = None # 初始化 model 变量
        try:
            model = init_detector(config_file, checkpoint_file, device=device)
            print(f"{model_name} 模型已加载到: {device}")
        except Exception as e_load:
            print(f"加载模型 {model_name} 失败: {e_load}")
            import traceback
            traceback.print_exc()
            continue

        # 初始化 DetLocalVisualizer
        # DetLocalVisualizer 是 MMDetection 3.x 中用于可视化的类
        visualizer = DetLocalVisualizer(
            vis_backends=[dict(type='LocalVisBackend')], # 指定后端，LocalVisBackend 用于本地保存文件
            save_dir=output_dir_model # 虽然 add_datasample 可以指定 out_file, 但 visualizer 也可设置默认保存目录
        )
        # 设置数据集元信息 (类别名称等)，这对于正确显示标签很重要
        # model.dataset_meta 是 MMDetection 3.x 模型通常包含的属性
        if hasattr(model, 'dataset_meta'):
            visualizer.dataset_meta = model.dataset_meta
            print(f"  已为 {model_name} 设置 Visualizer 的 dataset_meta。")
        else:
            # 对于 VOC 数据集，类别名称通常是固定的
            # 如果 dataset_meta 不可用，可以尝试手动设置 (如果需要显示类别名的话)
            # 例如: voc_classes = ('aeroplane', 'bicycle', ..., 'tvmonitor')
            # visualizer.dataset_meta = {'classes': voc_classes}
            print(f"  警告: 模型 {model_name} 没有 'dataset_meta' 属性。可视化标签可能不正确或缺失。")


        for img_idx, image_path in enumerate(image_paths):
            if not osp.exists(image_path):
                print(f"  图像路径未找到: {image_path}。跳过此图像。")
                continue

            base_filename = osp.basename(image_path)
            print(f"\n  正在对图像进行推理: {base_filename} (图像 {img_idx+1}/{len(image_paths)})")

            try:
                # 执行推理
                # 在 MMDetection 3.x 中，inference_detector 返回 DetDataSample 对象列表
                results_datasample_list = inference_detector(model, image_path)

                # 添加调试信息来检查实际返回的数据格式
                print(f"    推理结果类型: {type(results_datasample_list)}")
                if hasattr(results_datasample_list, '__len__'):
                    print(f"    推理结果长度: {len(results_datasample_list)}")
                
                # 读取图像为 numpy 数组 (BGR格式) 以供 visualizer 使用
                img_bgr_numpy = mmcv.imread(image_path)
                if img_bgr_numpy is None:
                    print(f"  错误: mmcv.imread 无法加载图像: {image_path}")
                    continue

                output_filename_base = os.path.splitext(base_filename)[0]
                # 输出文件名现在由 DetLocalVisualizer 内部根据 add_datasample 的 name 和 visualizer 的 save_dir 组合而成
                # 我们主要通过 out_file 参数精确控制它。
                output_filepath = osp.join(output_dir_model, f"{output_filename_base}_{model_name}_detection_segmentation.png")

                # 处理不同格式的推理结果
                data_sample_to_draw = None
                
                if isinstance(results_datasample_list, list):
                    # 如果是列表，取第一个元素
                    if len(results_datasample_list) > 0:
                        data_sample_to_draw = results_datasample_list[0]
                        print(f"    使用列表中的第一个DetDataSample: {type(data_sample_to_draw)}")
                    else:
                        print(f"    警告: 推理结果列表为空")
                        continue
                else:
                    # 如果不是列表，直接使用（可能是单个DetDataSample对象）
                    data_sample_to_draw = results_datasample_list
                    print(f"    使用单个DetDataSample对象: {type(data_sample_to_draw)}")

                if data_sample_to_draw is not None:
                    # 使用 DetLocalVisualizer 进行可视化
                    visualizer.add_datasample(
                        name=output_filename_base, # 通常用作窗口名或内部标识
                        image=img_bgr_numpy,       # 图像的 numpy 数组 (BGR 格式)
                        data_sample=data_sample_to_draw, # 推理结果 (DetDataSample 对象)
                        draw_gt=False,             # 我们只画预测结果
                        show=False,                # 不在屏幕上显示
                        wait_time=0,
                        pred_score_thr=score_threshold, # 分数阈值
                        out_file=output_filepath  # 直接指定输出文件完整路径
                    )
                    print(f"  可视化结果已保存到: {output_filepath}")
                else:
                    print(f"  警告: 无法获取有效的 DetDataSample 对象进行可视化")

            except Exception as e_img_proc:
                print(f"  处理图像 {image_path} 时发生错误 (模型: {model_name}): {e_img_proc}")
                import traceback
                traceback.print_exc()

    print(f"\n所有模型和图像处理完成。输出保存在 '{osp.abspath(output_dir_base)}' 下的各模型子目录中。")

if __name__ == '__main__':
    visualize_model_predictions()