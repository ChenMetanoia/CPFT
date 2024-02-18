import argparse
import torch
import os
from logging import getLogger
from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import init_seed, init_logger, get_trainer, set_color

from data.dataset import UniSRecDataset


def run_cpft(dataset, model, config_file_list, **kwargs):
    # configurations initialization
    model_name = model
    if f'props/{model_name}.yaml' not in config_file_list:
        config_file_list = [f'props/{model_name}.yaml'] + config_file_list
    print(config_file_list)

    # configurations initialization
    config = Config(model=model_name, dataset=dataset, config_file_list=config_file_list, config_dict=kwargs)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = UniSRecDataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    if model_name == 'FDSA':
        from model.fdsa_cpft import FDSA
        model = FDSA(config, train_data.dataset).to(config['device'])
    elif model_name == 'S3Rec':
        from model.s3rec_cpft import S3Rec
        model = S3Rec(config, train_data.dataset).to(config['device'])
    elif model_name == 'SASRec':
        from model.sasrec_cpft import SASRec
        model = SASRec(config, train_data.dataset).to(config['device'])
    elif model_name == 'UniSRec':
        from model.unisrec_cpft import UniSRec
        model = UniSRec(config, train_data.dataset).to(config['device'])
    else:
        raise NotImplementedError(f'The baseline [{model_name}] has not implemented yet.')
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    
    trainer = config_trainer(trainer, config)
    
    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )
    
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])
    
    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return config['model'], config['dataset'], {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }

def config_trainer(trainer, config):
    trainer.saved_model_file = os.path.join(trainer.checkpoint_dir, config['r'])
    trainer.cur_step = 0
    trainer.best_valid_score = 0
    checkpoint = torch.load(trainer.saved_model_file, map_location=trainer.device)
    trainer.model.load_state_dict(checkpoint["state_dict"])
    trainer.model.load_other_parameter(checkpoint.get("other_parameter"))
    trainer.model.alpha = config['alpha']
    trainer.model.beta = config['beta']
    trainer.model.gamma = config['gamma']
    trainer.model.cps = config['cps']
    trainer.model.cpd = config['cpd']
    trainer.model.dis_k = config['dis_k']
    trainer.model.calculate_by = config['calculate_by']
    trainer.model.distance_type = config['distance_type']
    return trainer
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SASRec', help='name of models')
    parser.add_argument("-d", type=str, default="Pantry", help="dataset name")
    parser.add_argument("-r", type=str, default="", help="regular-trained model path")
    parser.add_argument("--gpu_id", type=str, default="3", help="gpu id")
    parser.add_argument('--config_files', type=str, default='props/finetune.yaml', help='config files')
    parser.add_argument("--clip_grad_norm", action="store_true", help="clip grad norm")
    
    # cpft parameters
    parser.add_argument("--alpha", type=float, default=0.3, help="error rate")
    parser.add_argument("--beta", type=float, default=10, help="cps weight")
    parser.add_argument("--gamma", type=float, default=1, help="cpd weight")
    parser.add_argument(
        "--cps", action="store_true", help="calculate conformal prediction set size loss"
    )
    parser.add_argument(
        "--cpd", action="store_true", help="calculate conformal prediction set distance loss"
    )
    parser.add_argument(
        "--dis_k", type=int, default=10, help="topk items for distance loss"
    )
    parser.add_argument(
        "--calculate_by",
        type=str,
        default="cosine",
        help="calculate distance by [cosine, euclidean]",
    )
    parser.add_argument(
        "--distance_type",
        type=str,
        default="min",
        help="measure distance by cloest, average or farthest item [min, avg, max]",
    )

    args, unparsed = parser.parse_known_args()
    config_dict = vars(args)
    run_cpft(args.m, args.d, args.r, args.config_files.strip().split(' '), **config_dict)
    
