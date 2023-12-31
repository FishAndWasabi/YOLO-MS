onnx_config = dict(type='onnx',
                   export_params=True,
                   keep_initializers_as_inputs=False,
                   opset_version=11,
                   save_file='end2end.onnx',
                   input_names=['input'],
                   output_names=['feat0', 'feat1', 'feat2'],
                   input_shape=[320, 320],
                   optimize=True)
codebase_config = dict(type='mmyolo',
                       task='ObjectDetection',
                       model_type='rknn',
                       post_processing=dict(score_threshold=0.05,
                                            confidence_threshold=0.005,
                                            iou_threshold=0.5,
                                            max_output_boxes_per_class=200,
                                            pre_top_k=5000,
                                            keep_top_k=100,
                                            background_label_id=-1),
                       module=['mmyolo.deploy', 'mmyolo', 'yoloms'])
backend_config = dict(type='rknn',
                      common_config=dict(target_platform='rv1126',
                                         optimization_level=1),
                      quantization_config=dict(do_quantization=False,
                                               dataset=None),
                      input_size_list=[[3, 320, 320]])
