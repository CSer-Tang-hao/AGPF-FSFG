import os
import time

import numpy as np
import torch
import torch.optim

import backbone
import configs
import utils
from backbone_apcnn import APCNN
from data.datamgr import SetDataManager
from io_utils import model_dict, parse_args, get_resume_file
from methods.matchingnet import MatchingNet
from methods.protonet import ProtoNet
from methods.relationnet import RelationNet


def train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params):
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
    else:
        raise ValueError('Unknown optimization, please define by yourself')

    max_acc = 0

    for epoch in range(start_epoch, stop_epoch):
        start_time = time.time()
        model.train()
        model.train_loop(epoch, base_loader, optimizer)  # model are called by reference, no need to return
        model.eval()

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        acc = model.test_loop(val_loader)
        if acc > max_acc:
            print("best model! save...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

        if (epoch % params.save_freq == 0) or (epoch == stop_epoch - 1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
        end_time = time.time()
        print("epoch time {} s".format(end_time - start_time))

    return model


def main():
    utils.setup_seed(1)
    print('---------- train begin-------------')
    np.random.seed(10)
    params = parse_args('train')

    base_file = configs.data_dir[params.dataset] + 'base.json'
    val_file = configs.data_dir[params.dataset] + 'val.json'

    image_size = 84

    optimization = 'Adam'

    if params.stop_epoch == -1:
        params.stop_epoch = 120

    if params.method in ['protonet', 'matchingnet', 'relationnet', 'relationnet_softmax']:
        n_query = max(1, int(
            params.n_query * params.test_n_way / params.train_n_way))  # if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small

        train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
        base_datamgr = SetDataManager(image_size, n_query=n_query, n_eposide=500, **train_few_shot_params)
        base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)

        test_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
        val_datamgr = SetDataManager(image_size, n_query=n_query, n_eposide=500, **test_few_shot_params)
        val_loader = val_datamgr.get_data_loader(val_file, aug=False)
        # a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor

        if params.method == 'protonet':
            model = ProtoNet(model_dict[params.model], **train_few_shot_params)
        elif params.method == 'matchingnet':
            model = MatchingNet(model_dict[params.model], **train_few_shot_params)
        elif params.method in ['relationnet', 'relationnet_softmax']:
            if params.model == 'Conv4':
                feature_model = backbone.Conv4NP
            else:
                feature_model = lambda: model_dict[params.model](flatten=False)
            loss_type = 'mse' if params.method == 'relationnet' else 'softmax'

            model = RelationNet(feature_model, loss_type=loss_type, **train_few_shot_params)
    else:
        raise ValueError('Unknown method')
    if params.apcnn:
        model.feature = APCNN(model.feature, params.num_classes)
    else:
        raise ValueError('params.apcnn must be true')

    model = model.cuda()
    print(model)
    # params.checkpoint_dir = './%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    params.checkpoint_dir = os.path.join(configs.save_dir, 'checkpoints', params.dataset,
                                         '{:}_{:}'.format(params.model, params.method))
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    params.checkpoint_dir += '_%dway_%dshot' % (params.train_n_way, params.n_shot)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    if params.resume:
        resume_file = get_resume_file(params.checkpoint_dir)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch'] + 1
            model.load_state_dict(tmp['state'])

    print(params)
    model = train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params)


if __name__ == '__main__':
    main()
