from first_stage_train.model import CodeBert_Seq2Seq

# 加载微调过的模型
model = CodeBert_Seq2Seq(codebert_path = 'D:\\syh\\models\\codebert-base', decoder_layers = 6, fix_encoder = True, beam_size = 10,
                         max_source_length = 64, max_target_length = 64, load_model_path = './result1/pytorch_model.bin',
                         l2_norm=True, fusion=True)

# 模型测试
model.test(test_filename = 'data/first_stage_data/test.csv', test_batch_size = 16, output_dir = './result1')

# # 模型推理单条数据
# comment = model.predict(source = 'find . -exec printf %s\0 {} ;', similarity='find . | xargs -i{} printf %s%s\n {} {}')
# print(comment)