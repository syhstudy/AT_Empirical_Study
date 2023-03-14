from first_stage_train.model import CodeBert_Seq2Seq

# 初始化模型
model = CodeBert_Seq2Seq(codebert_path = 'D:\\syh\\models\\codebert-base', decoder_layers = 6, fix_encoder = False, beam_size = 10,
                         max_source_length = 32, max_target_length = 64, load_model_path = None)

# 模型训练
model.train(train_filename ='../data/pre_shell/train.csv', train_batch_size = 64, num_train_epochs = 50, learning_rate = 2e-4,
            do_eval = True, dev_filename ='../data/pre_shell/valid.csv', eval_batch_size = 64, output_dir ='../pretrained_model')

# 加载微调过的模型
model = CodeBert_Seq2Seq(codebert_path = 'D:\\syh\\models\\codebert-base', decoder_layers = 6, fix_encoder = True, beam_size = 10,
                         max_source_length = 32, max_target_length = 64, load_model_path = '../pretrained_model/pytorch_model.bin')

# 模型测试
model.test(test_filename ='../data/pre_shell/test.csv', test_batch_size = 16, output_dir ='../result/CodeBert')