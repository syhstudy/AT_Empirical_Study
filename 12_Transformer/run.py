from model import Transformer_Seq2Seq

# 初始化模型
model = Transformer_Seq2Seq(codebert_path = 'D:\\models\\roberta-base', encoder_layers = 6, decoder_layers = 6, beam_size = 10,
                         max_source_length = 256, max_target_length = 256, load_model_path = None)

# 模型训练
model.train(train_filename = 'data/train.json', train_batch_size = 32, num_train_epochs = 20, learning_rate = 5e-4,
              do_eval = True, dev_filename = 'data/dev.json', eval_batch_size = 32, output_dir = 'valid_output')

# 加载微调过的模型
# model = CodeBert_Seq2Seq(codebert_path = 'D:\\new_idea\\Final\\model\\codebert', decoder_layers = 6, fix_encoder = True, beam_size = 10,
#                          max_source_length = 256, max_target_length = 32, load_model_path = './valid_output/checkpoint-last/pytorch_model.bin')

# # 模型测试
# model.test(test_filename = 'data/dev.json', test_batch_size = 32, output_dir = 'valid_output')
#
# # 模型推理
# comment = model.predict(source = "enable/disable gps concode_field_sep Context context concode_field_sep void reboot concode_elem_sep boolean getGPS")
# print(comment)