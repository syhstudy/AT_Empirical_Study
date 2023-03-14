import datetime
import logging
from PLM import PLM_model
from utils import set_seed

set_seed(42)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

model_dict = {
    'codet5' : 'D:\model\codet5-base',
}

model_type = 'codet5'

# 初始化模型
model = PLM_model(model_type=model_type, model_name_or_path=model_dict[model_type], load_model_path=None,
                  beam_size=10, max_source_length=64, max_target_length=128)

start = datetime.datetime.now()

# 模型训练
model.train(train_filename = 'data/Bash_train.csv', train_batch_size = 32, learning_rate = 2e-4,
            num_train_epochs = 50, early_stop = 5, do_eval=True, eval_filename='data/Bash_valid.csv',
            eval_batch_size=12, output_dir='valid_output/'+model_type+'/', do_eval_bleu=True)

end = datetime.datetime.now()
print(end-start)

# 加载微调过后的模型参数
model = PLM_model(model_type=model_type, model_name_or_path=model_dict[model_type], beam_size=10,
                  max_source_length=64, max_target_length=128,
                  load_model_path='valid_output/'+model_type+'/checkpoint-best-bleu/pytorch_model.bin')

model.test(batch_size=32, filename='data/Bash_test.csv', output_dir='test_output/'+model_type+'/')
# model.test(batch_size=32, filename='data/big_test.csv', output_dir='test_output/')

# comment = model.predict(source = "echo $(/usr/sbin/arp $(hostname) | awk -F[()] {print $2})")
# print(comment)