import torch
from torch.autograd import Variable
import os

import h5py
import torch
from torch.autograd import Variable

import backbone
import configs
import utils
from backbone_apcnn import APCNN
from data.datamgr import SimpleDataManager
from io_utils import model_dict, parse_args, get_best_file, get_assigned_file


# from backbone_apcnn_singlescale import APCNN,DoNothing
def save_features(model, data_loader, outfile):
    f = h5py.File(outfile, 'w')
    max_count = len(data_loader) * data_loader.batch_size
    all_labels = f.create_dataset('all_labels', (max_count,), dtype='i')
    all_feats = None
    count = 0
    for i, (x, y) in enumerate(data_loader):
        if i % 10 == 0:
            print('{:d}/{:d}'.format(i, len(data_loader)))
        x = x.cuda()
        x_var = Variable(x)
        feats, traditional_score = model(x_var)
        if all_feats is None:
            all_feats = f.create_dataset('all_feats', [max_count] + list(feats.size()[1:]), dtype='f')
        all_feats[count:count + feats.size(0)] = feats.data.cpu().numpy()
        all_labels[count:count + feats.size(0)] = y.cpu().numpy()
        count = count + feats.size(0)

    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count

    f.close()


if __name__ == '__main__':
    utils.setup_seed(2)
    params = parse_args('save_features')
    image_size = 84
    split = params.split

    loadfile = configs.data_dir[params.dataset] + split + '.json'
    checkpoint_dir = os.path.join(configs.save_dir, 'checkpoints', params.dataset,
                                  '{:}_{:}'.format(params.model, params.method))
    # checkpoint_dir = './%s/checkpoints/%s/%s_%s' %()
    if params.train_aug:
        checkpoint_dir += '_aug'
    checkpoint_dir += '_%dway_%dshot' % (params.train_n_way, params.n_shot)

    if params.save_iter != -1:
        modelfile = get_assigned_file(checkpoint_dir, params.save_iter)
    else:
        modelfile = get_best_file(checkpoint_dir)

    if params.save_iter != -1:
        outfile = os.path.join(checkpoint_dir.replace("checkpoints", "features"),
                               split + "_" + str(params.save_iter) + ".hdf5")
    else:
        outfile = os.path.join(checkpoint_dir.replace("checkpoints", "features"), split + ".hdf5")

    datamgr = SimpleDataManager(image_size, batch_size=64)
    data_loader = datamgr.get_data_loader(loadfile, aug=False)

    if params.method in ['relationnet', 'relationnet_softmax']:
        if params.model == 'Conv4':
            model = backbone.Conv4NP()
        else:
            model = model_dict[params.model](flatten=False)
    else:
        model = model_dict[params.model]()
    if params.apcnn:
        model = APCNN(model, params.num_classes)
    else:
        raise ValueError('params.apcnn must be true')
    model = model.cuda()
    tmp = torch.load(modelfile)
    state = tmp['state']
    state_keys = list(state.keys())
    for i, key in enumerate(state_keys):
        if "feature." in key:
            newkey = key.replace("feature.",
                                 "")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
            state[newkey] = state.pop(key)
        else:
            state.pop(key)

    model.load_state_dict(state)
    model.eval()

    dirname = os.path.dirname(outfile)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    with torch.no_grad():
        save_features(model, data_loader, outfile)
