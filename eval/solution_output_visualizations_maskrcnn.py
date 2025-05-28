import mmcv
import torch
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmengine.dataset import Compose
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp

# 更新后的 prepare_data_for_rpn 函数，包含关键的调试信息
def prepare_data_for_rpn(img_path, model_cfg, device):
    """
    Loads an image, processes it through the test pipeline, and prepares
    the image tensor and image metadata for RPN input.
    """
    print(f"DEBUG: prepare_data_for_rpn - 开始处理图像: {img_path}")

    pipeline_cfg = None
    if 'test_dataloader' in model_cfg and hasattr(model_cfg.test_dataloader, 'dataset') and hasattr(model_cfg.test_dataloader.dataset, 'pipeline'):
        print("DEBUG: prepare_data_for_rpn - 使用 MMDetection 3.x 风格的 pipeline 路径 (cfg.test_dataloader.dataset.pipeline)")
        pipeline_cfg = model_cfg.test_dataloader.dataset.pipeline
    elif 'data' in model_cfg and 'test' in model_cfg.data and hasattr(model_cfg.data.test, 'pipeline'):
        print("DEBUG: prepare_data_for_rpn - 使用 MMDetection 2.x 风格的 pipeline 路径 (cfg.data.test.pipeline)")
        pipeline_cfg = model_cfg.data.test.pipeline
    else:
        print(f"DEBUG: prepare_data_for_rpn - model_cfg 内容: {model_cfg}")
        raise ValueError("在模型配置中找不到测试数据处理流程 (test pipeline)。请检查 cfg.test_dataloader.dataset.pipeline (MMDet3) 或 cfg.data.test.pipeline (MMDet2)。")

    print(f"DEBUG: prepare_data_for_rpn - 选择的 pipeline_cfg: {pipeline_cfg}")

    if pipeline_cfg[0]['type'] == 'LoadImageFromFile':
        data = {'img_path': img_path}
        print(f"DEBUG: prepare_data_for_rpn - 初始化 'data' for LoadImageFromFile: {data}")
    else:
        img_bgr_loaded = mmcv.imread(img_path)
        if img_bgr_loaded is None:
            raise IOError(f"prepare_data_for_rpn - mmcv.imread 无法加载图像: {img_path}")
        data = {
            'img': img_bgr_loaded, 'img_shape': img_bgr_loaded.shape, 'ori_shape': img_bgr_loaded.shape,
            'filename': img_path, 'ori_filename': img_path, 'pad_shape': img_bgr_loaded.shape,
            'scale_factor': np.array([1., 1., 1., 1.], dtype=np.float32),
        }
        print(f"DEBUG: prepare_data_for_rpn - 初始化 'data' (图像已预加载): keys={list(data.keys())}")

    test_pipeline = Compose(pipeline_cfg)
    processed_data = None
    try:
        print("DEBUG: prepare_data_for_rpn - 即将执行 test_pipeline(data)...")
        processed_data = test_pipeline(data)
        print("DEBUG: prepare_data_for_rpn - test_pipeline(data) 执行完毕。")
    except Exception as e:
        print(f"DEBUG: prepare_data_for_rpn - pipeline 执行期间发生错误: {e}")
        print(f"DEBUG: prepare_data_for_rpn - 送入 pipeline 的初始 'data': {data}")
        import traceback
        traceback.print_exc()
        raise

    if processed_data is None:
        raise ValueError("prepare_data_for_rpn - 数据 pipeline 返回 None。请检查您的 pipeline 转换中是否有错误。")

    print(f"DEBUG: prepare_data_for_rpn - pipeline 返回的 processed_data 的键: {list(processed_data.keys()) if isinstance(processed_data, dict) else '不是一个字典'}")
    # (Previous debug prints for 'img' key can be kept or removed for brevity now that 'inputs' is confirmed)

    img_tensor_candidate = None
    if isinstance(processed_data, dict) and 'inputs' in processed_data and torch.is_tensor(processed_data['inputs']):
        print("DEBUG: prepare_data_for_rpn - 找到 'inputs' 键且其值为张量。将使用 'inputs' 作为图像张量。")
        img_tensor_candidate = processed_data['inputs']
    elif isinstance(processed_data, dict) and 'img' in processed_data: # Fallback if 'inputs' isn't there but 'img' is
        print("DEBUG: prepare_data_for_rpn - 找到 'img' 键。将使用 'img' 作为图像张量。")
        img_tensor_candidate = processed_data['img']
    else:
        available_keys_str = str(list(processed_data.keys()) if isinstance(processed_data, dict) else '不是一个字典')
        raise KeyError(f"prepare_data_for_rpn - 数据 pipeline 未生成 'inputs' 或 'img' 键。可用键: {available_keys_str}。请检查您的测试 pipeline 配置。")

    if isinstance(img_tensor_candidate, list):
        if not img_tensor_candidate:
            raise ValueError("prepare_data_for_rpn - Pipeline 为图像张量候选键生成了一个空列表。")
        img_tensor_from_list = img_tensor_candidate[0]
    else:
        img_tensor_from_list = img_tensor_candidate

    if not torch.is_tensor(img_tensor_from_list):
        raise TypeError(f"prepare_data_for_rpn - 期望图像张量是 PyTorch 张量或张量列表，但得到 {type(img_tensor_from_list)}")

    final_img_tensor = img_tensor_from_list # 'inputs' is already batched by PackDetInputs typically
    if final_img_tensor.ndim == 3: # If it's C, H, W, add batch dimension
        final_img_tensor = final_img_tensor.unsqueeze(0)
    final_img_tensor = final_img_tensor.to(device)
    print(f"DEBUG: prepare_data_for_rpn - final_img_tensor shape: {final_img_tensor.shape}")

    img_metas_list_of_dicts = None
    if isinstance(processed_data, dict) and 'data_samples' in processed_data:
        data_samples_content = processed_data['data_samples']
        print(f"DEBUG: prepare_data_for_rpn - 'data_samples' 的类型: {type(data_samples_content)}")

        # Determine if data_samples_content is a list of DetDataSample or a single DetDataSample
        current_sample = None
        if isinstance(data_samples_content, list):
            if not data_samples_content:
                raise ValueError("prepare_data_for_rpn - 'data_samples' 是一个空列表。")
            current_sample = data_samples_content[0]
            print("DEBUG: prepare_data_for_rpn - 'data_samples' 是一个列表，取第一个元素。")
        else:
            # Assuming it might be a single DetDataSample object if not a list
            # This case might be less common for PackDetInputs output but good to be aware
            current_sample = data_samples_content
            print("DEBUG: prepare_data_for_rpn - 'data_samples' 不是列表，直接使用。")
        
        if not hasattr(current_sample, 'metainfo'):
             raise AttributeError("prepare_data_for_rpn - DetDataSample 对象没有 'metainfo' 属性。")

        sample_metainfo = current_sample.metainfo
        print(f"DEBUG: prepare_data_for_rpn - 从 DetDataSample.metainfo 提取的 sample_metainfo 键: {list(sample_metainfo.keys())}")

        img_metas_single_dict = {
            'filename': sample_metainfo.get('img_path', img_path),
            'ori_filename': sample_metainfo.get('img_path', img_path), # Use img_path as fallback
            'ori_shape': sample_metainfo.get('ori_shape'),
            'img_shape': sample_metainfo.get('img_shape'),
            'pad_shape': sample_metainfo.get('pad_shape', sample_metainfo.get('img_shape')),
            'scale_factor': sample_metainfo.get('scale_factor'), # RPNHead usually expects this
            'flip': sample_metainfo.get('flip', False),
            'flip_direction': sample_metainfo.get('flip_direction', None)
        }
        # Ensure critical keys RPN might need are present
        if img_metas_single_dict['ori_shape'] is None and 'img_shape' in sample_metainfo:
            img_metas_single_dict['ori_shape'] = sample_metainfo['img_shape']
        if img_metas_single_dict['img_shape'] is None and final_img_tensor.ndim >=3 :
             img_metas_single_dict['img_shape'] = final_img_tensor.shape[-2:] # H, W from tensor itself (after batch)
        if img_metas_single_dict['scale_factor'] is None:
            # RPN proposals need scale_factor to rescale back to original image size.
            # If not directly available, try to compute or use a default, though this might be risky.
            # Example: if img_shape and ori_shape are available:
            # ih, iw = sample_metainfo.get('img_shape')
            # oh, ow = sample_metainfo.get('ori_shape')
            # scale_x = iw / ow
            # scale_y = ih / oh
            # img_metas_single_dict['scale_factor'] = np.array([scale_x, scale_y, scale_x, scale_y], dtype=np.float32)
            print("WARNING: prepare_data_for_rpn - 'scale_factor' not found in metainfo. RPN rescaling might be incorrect. Using default [1,1,1,1].")
            img_metas_single_dict['scale_factor'] = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)


        img_metas_list_of_dicts = [img_metas_single_dict]
    elif isinstance(processed_data, dict) and 'img_metas' in processed_data: # Fallback for older MMDetection 2.x style if somehow encountered
        img_metas_dc = processed_data['img_metas']
        if isinstance(img_metas_dc, list):
            img_metas_single_dict = img_metas_dc[0].data
        else:
            img_metas_single_dict = img_metas_dc.data
        img_metas_list_of_dicts = [img_metas_single_dict]
    else:
        available_keys_str = str(list(processed_data.keys()) if isinstance(processed_data, dict) else '不是一个字典')
        raise KeyError(f"prepare_data_for_rpn - 数据 pipeline 未生成 'data_samples' 或 'img_metas' 键。可用键: {available_keys_str}。请检查 pipeline 配置。")

    print(f"DEBUG: prepare_data_for_rpn - img_metas_list_of_dicts[0] keys: {list(img_metas_list_of_dicts[0].keys())}")
    original_img_bgr = mmcv.imread(img_path)
    print(f"DEBUG: prepare_data_for_rpn - 成功完成图像处理: {img_path}")
    return original_img_bgr, final_img_tensor, img_metas_list_of_dicts


def visualize_mask_rcnn_predictions():
    # --- 用户配置 ---
    mask_rcnn_config_file = '/data/jiangzishang/lty/作业/张力/mmdetection/configs/mask_rcnn/mask-rcnn_r50_fpn_1x_voc.py'
    mask_rcnn_checkpoint_file = '/data/jiangzishang/lty/作业/张力/mmdetection/work_dirs/mask_rcnn_voc/epoch_50.pth'

    image_paths = [
        '/data/jiangzishang/lty/作业/张力/mmdetection/data/coco/VOC2007/JPEGImages/000042.jpg',
        '/data/jiangzishang/lty/作业/张力/mmdetection/data/coco/VOC2007/JPEGImages/000061.jpg',
        '/data/jiangzishang/lty/作业/张力/mmdetection/data/coco/VOC2007/JPEGImages/000100.jpg',
        '/data/jiangzishang/lty/作业/张力/mmdetection/data/coco/VOC2007/JPEGImages/000101.jpg',
    ]
    output_dir = './output_visualizations_maskrcnn/'
    # --- 配置结束 ---

    if not image_paths:
        print("错误: `image_paths` 列表为空。请提供图像路径。")
        return

    # 使用 os.makedirs 替代 mmcv.mkdir_or_exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录 '{output_dir}' 已确保存在。")

    try:
        register_all_modules(init_default_scope=False) # MMDetection 2.x 风格
        print("MMDetection模块已使用 init_default_scope=False 注册。")
    except Exception:
        try:
            register_all_modules(init_default_scope=True) # MMDetection 3.x 风格
            print("MMDetection模块已使用 init_default_scope=True 注册。")
        except Exception as e:
            print(f"注册MMDetection模块时出错: {e}")
            return

    print("正在加载Mask R-CNN模型...")
    try:
        model = init_detector(mask_rcnn_config_file, mask_rcnn_checkpoint_file, device='cuda:0') # 或 'cpu'
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    cfg = model.cfg # cfg 就是 model.cfg
    device = next(model.parameters()).device
    print(f"模型已加载到: {device}")

    for i, image_path in enumerate(image_paths[:4]):
        if not osp.exists(image_path):
            print(f"图像路径未找到: {image_path}。跳过此图像。")
            continue

        base_filename = osp.basename(image_path)
        print(f"\n--- 正在处理图像 {i+1}/{len(image_paths)}: {base_filename} ---")

        try:
            # 1. 准备数据
            #    现在 prepare_data_for_rpn 包含详细的调试信息
            original_img_bgr, img_tensor, img_metas = prepare_data_for_rpn(image_path, cfg, device)

            # 2. 提取RPN proposals
            print("正在提取RPN proposals...")
            model.eval() # 确保模型在评估模式
            with torch.no_grad():
                features = model.extract_feat(img_tensor)
                proposals_np = np.empty((0,5)) # 初始化以防没有RPN头
                if hasattr(model, 'rpn_head') and model.rpn_head is not None:
                    try:
                        # 首先尝试使用 forward 方法获取 RPN 输出
                        rpn_out = model.rpn_head.forward(features)
                        if isinstance(rpn_out, tuple) and len(rpn_out) == 2:
                            cls_scores, bbox_preds = rpn_out
                            print(f"DEBUG: RPN cls_scores shape: {[s.shape for s in cls_scores]}")
                            print(f"DEBUG: RPN bbox_preds shape: {[b.shape for b in bbox_preds]}")
                            
                            # 使用 MMDetection 3.x 的新API - predict_by_feat
                            try:
                                # 尝试使用 predict_by_feat，但不传入 cfg 参数
                                proposal_list = model.rpn_head.predict_by_feat(
                                    cls_scores, bbox_preds, 
                                    batch_img_metas=img_metas,
                                    rescale=True
                                )
                                if proposal_list and len(proposal_list) > 0:
                                    # 在MMDetection 3.x中，返回的是InstanceData对象
                                    if hasattr(proposal_list[0], 'bboxes'):
                                        proposals_tensor = proposal_list[0].bboxes
                                        proposals_np = proposals_tensor.cpu().numpy()
                                    else:
                                        print("警告: RPN proposals 格式不符合预期。")
                                else:
                                    print("警告: RPN head predict_by_feat 返回空列表。")
                            except Exception as e2:
                                print(f"使用 predict_by_feat 失败: {e2}")
                                # 回退方案：直接使用 loss_and_predict 或其他方法
                                try:
                                    # 尝试使用模型的整体predict方法
                                    with torch.no_grad():
                                        # 构建inputs字典
                                        batch_data_samples = [sample for sample in [None]]  # 用于inference
                                        batch_inputs = img_tensor
                                        
                                        # 使用模型的predict方法
                                        results = model.predict(batch_inputs, img_metas, rescale=True)
                                        if results and len(results) > 0 and hasattr(results[0], 'proposals'):
                                            proposals_tensor = results[0].proposals.bboxes
                                            proposals_np = proposals_tensor.cpu().numpy()
                                        else:
                                            print("警告: 无法从模型predict结果中提取proposals。")
                                except Exception as e3:
                                    print(f"使用模型predict方法也失败: {e3}")
                        else:
                            print(f"警告: RPN head forward 返回意外的输出格式: {type(rpn_out)}")
                    except Exception as e:
                        print(f"RPN 处理过程中出错: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print("警告: 此模型没有RPN头 (rpn_head)，无法提取proposals。")

            print(f"提取到 {proposals_np.shape[0]} 个 RPN proposals。")

            # 可视化RPN proposals
            if proposals_np.size > 0:
                # ... (可视化代码部分，与之前相同，此处省略以保持简洁) ...
                img_for_proposals_vis = original_img_bgr.copy()
                if proposals_np.shape[1] == 4: # 如果分数缺失
                    proposals_np = np.concatenate([proposals_np, np.ones((proposals_np.shape[0], 1))], axis=1)
                
                proposal_scores = proposals_np[:, 4]
                sorted_indices = np.argsort(proposal_scores)[::-1]
                num_proposals_to_show = min(100, proposals_np.shape[0])
                top_k_proposals = proposals_np[sorted_indices[:num_proposals_to_show]]

                plt.figure(figsize=(12, 8))
                plt.imshow(mmcv.bgr2rgb(img_for_proposals_vis))
                ax = plt.gca()
                for prop_idx in range(top_k_proposals.shape[0]):
                    box = top_k_proposals[prop_idx, :4]
                    ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                               fill=False, edgecolor='red', linewidth=0.8))
                plt.title(f"Image: {base_filename}\nRPN Proposals (Top {num_proposals_to_show} of {proposals_np.shape[0]})")
                plt.axis('off')
                proposal_out_file = osp.join(output_dir, f'{os.path.splitext(base_filename)[0]}_rpn_proposals.png')
                plt.savefig(proposal_out_file)
                print(f"RPN proposal可视化结果已保存到: {proposal_out_file}")
                plt.close()
            else:
                print("没有RPN proposals可供可视化。")


            # 3. 获取并可视化最终预测结果
            print("正在获取最终预测结果...")
            final_pred_results = inference_detector(model, image_path)

            # 使用MMDetection 3.x的新可视化API
            from mmdet.visualization import DetLocalVisualizer
            visualizer = DetLocalVisualizer()
            
            # 读取图像
            img = mmcv.imread(image_path)
            img = mmcv.imconvert(img, 'bgr', 'rgb')
            
            final_pred_out_file = osp.join(output_dir, f'{os.path.splitext(base_filename)[0]}_final_predictions.png')
            
            # 使用新的可视化API
            visualizer.add_datasample(
                name=f"Final Predictions - {base_filename}",
                image=img,
                data_sample=final_pred_results,
                draw_gt=False,
                show=False,
                out_file=final_pred_out_file,
                pred_score_thr=0.3
            )
            print(f"最终预测可视化结果已保存到: {final_pred_out_file}")

            num_final_dets = 0
            bbox_results = final_pred_results[0] if isinstance(final_pred_results, tuple) else final_pred_results
            for per_class_bboxes in bbox_results:
                if per_class_bboxes.ndim == 2 and per_class_bboxes.shape[1] == 5:
                    num_final_dets += np.sum(per_class_bboxes[:, 4] >= 0.3)
            print(f"最终检测到的目标数量 (score >= 0.3): {num_final_dets}")

        except Exception as e:
            print(f"处理图像 {image_path} 时发生主循环错误: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n所有处理完成。可视化结果保存在目录: {osp.abspath(output_dir)}")

if __name__ == '__main__':
    visualize_mask_rcnn_predictions()