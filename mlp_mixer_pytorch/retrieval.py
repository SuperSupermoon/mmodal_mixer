"""
Downstream task: Retrieval (IR & TR)
Label conditioned retrieval task
Metric
 - Recall@1,5,10
 - Preecision@K
 - Hit@K
 - MRR_score
"""
import os
import json
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from fuzzywuzzy import fuzz
from datetime import datetime
from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import wandb
from torch.optim import SGD, AdamW, Adam

from transformers import AutoConfig
from transformers import BertConfig, BertModel, BertTokenizer
# from models.cxrbert_origin import CXRBERT

from functools import partial
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat

import torch
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def truncate_txt(txt_tokens, max_seq_len):
    while True:
        if len(txt_tokens) <= max_seq_len:
            break
        else:
            txt_tokens.pop()


class CXR_Retrieval_Dataset(Dataset):
    def __init__(self, data_path, tokenizer, transforms, args, is_train=True):
        self.args = args
        self.data_dir = os.path.dirname(data_path)
        self.data = [json.loads(line) for line in open(data_path)]

        self.num_image_embeds = args.num_image_embeds
        self.seq_len = args.seq_len
        self.transforms = transforms

        self.is_train = is_train
        self.tokenizer = tokenizer  # tokenizer = BertTokenizer.from_pretrained('bert-based-uncased').tokenize
        self.BertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.vocab_stoi = self.BertTokenizer.vocab
        self.vocab_len = len(self.vocab_stoi)  # 30522


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        if self.is_train:
            if args.MIMIC_dset:
                study_id, split, label, txt, img = self.data[idx].keys()
            else:
                study_id, label, txt, img = self.data[idx].keys()

            d_label = self.data[idx][label]
            d_txt = self.data[idx][txt]
            d_img = self.data[idx][img]

            if self.args.label_conditioned:
                for itr in range(300):
                    random_label, random_txt, random_img = self.get_random_line(idx)
                    if fuzz.token_sort_ratio(d_label, random_label) != 100:
                        if random.random() > 0.5:
                            neg_img, neg_txt = random_img, d_txt
                            break
                        else:
                            neg_img, neg_txt = d_img, random_txt
                            break
                    else:
                        pass
            else:
                _, random_txt, random_img = self.get_random_line(idx)
                if random.random() > 0.5:
                    neg_img, neg_txt = random_img, d_txt
                else:
                    neg_img, neg_txt = d_img, random_txt

            pair_pos = self.data_processing(d_txt, d_img)
            pair_neg = self.data_processing(neg_txt, neg_img)

            example = tuple(list(pair_pos) + [1] + list(pair_neg) + [0])

            return idx, example

        else:
            """
            {"id": "s57495790", "split": "Valid", "label": "'Lung Opacity'", "is_aligned": [1], "T_label": "'Lung Opacity'", 
            "txt": "Single portable view of the chest. Lower lung volumes seen ...", 
            "img": "/home/mimic-cxr/dataset/image_preprocessing/re_512_3ch/Valid/s57495790.jpg"}
            T_label is changed to T_id
            """
            if args.MIMIC_dset:
                study_id, split, label, is_aligned, r_id, txt, img = self.data[idx].keys()
            else:
                study_id, label, is_aligned, r_id, txt, img = self.data[idx].keys()
            txt = self.data[idx][txt]
            img = self.data[idx][img]
            label = self.data[idx][is_aligned]  # 1(Aligned), 0(Not aligned)

            sample = self.data_processing(txt, img)
            example = tuple(list(sample) + label + [idx])
            return example


    def get_random_line(self, idx):
        rand_idx = list(range(0, idx)) + list(range(idx + 1, len(self.data)))
        rand_num = random.choice(rand_idx)
        label = self.data[rand_num]['label']
        txt = self.data[rand_num]['text']
        img = self.data[rand_num]['img']
        return label, txt, img

    def data_processing(self, origin_txt, img_path):
        if args.CXRBERT:
            if self.args.img_channel == 3:
                image = Image.open(os.path.join(self.data_dir, img_path))
            elif self.args.img_channel == 1:
                image = Image.open(os.path.join(self.data_dir, img_path))
                image = transforms.Grayscale(num_output_channels=3)(image)

            print("image", image.shape)
            image = self.transforms(image)


            tokenized_sentence = self.tokenizer(origin_txt)  # ['i','ate','an','apple'], no special token
            

            truncate_txt(tokenized_sentence, self.seq_len)


            encoded_sentence = [self.vocab_stoi[w] if w in self.vocab_stoi else self.vocab_stoi["[UNK]"]
                                    for w in tokenized_sentence]  # [178, 8756, 1126, 12075]

            input_ids = encoded_sentence + [self.vocab_stoi["[SEP]"]]

            attn_masks_t = [1] * len(input_ids)
            attn_masks_i = [1] * (self.args.num_image_embeds + 2)  # [CLS]

            padding = [self.vocab_stoi["[PAD]"] for _ in range(self.seq_len - len(input_ids) + 1)]  # 0, [CLS]

            input_ids.extend(padding)
            attn_masks_t.extend(padding)

            attn_masks = attn_masks_i + attn_masks_t

            segment = [1 for _ in range(self.seq_len + 1)]

            cls_tok = [self.vocab_stoi["[CLS]"]]
            cls_tok = torch.tensor(cls_tok)

            sep_tok = [self.vocab_stoi["[SEP]"]]
            sep_tok = torch.tensor(sep_tok)

            input_ids = torch.tensor(input_ids)


            attn_masks = torch.tensor(attn_masks)
            segment = torch.tensor(segment)

            return cls_tok, input_ids, attn_masks, image, segment, sep_tok

        else:
            tokenized_sentence = self.tokenizer(origin_txt)  # ['i','ate','an','apple'], no special token

            truncate_txt(tokenized_sentence, self.seq_len)

            encoded_sentence = [self.vocab_stoi[w] if w in self.vocab_stoi else self.vocab_stoi["[UNK]"]
                                for w in tokenized_sentence]  # [178, 8756, 1126, 12075]

            input_ids = [self.vocab_stoi["[CLS]"]] + encoded_sentence + [self.vocab_stoi["[SEP]"]]


            padding = [self.vocab_stoi["[PAD]"] for _ in range(self.seq_len - len(input_ids) + 2)]

            input_ids.extend(padding)
            cls_tok = [self.vocab_stoi["[CLS]"]]
            cls_tok = torch.tensor(cls_tok)
            segment = [1 for _ in range(self.seq_len + 2)]

            input_ids = torch.tensor(input_ids)
            segment = torch.tensor(segment)


            if self.args.img_channel == 3:
                image = Image.open(os.path.join(self.data_dir, img_path))
            elif self.args.img_channel == 1:
                image = Image.open(os.path.join(self.data_dir, img_path))
                image = transforms.Grayscale(num_output_channels=3)(image)

            image = self.transforms(image)

            return input_ids, cls_tok, segment, image


def compute_ranks(args, results, labels, idx_lst):
    labels = np.array(labels)
    # print('len_ labels, result, idx_lst:', len(labels), len(results), len(idx_lst))
    similarities = np.array([results[i] for i in range(len(idx_lst))])

    num_txt_per_img = args.eval_len_size

    labels = np.reshape(labels, [-1, num_txt_per_img])
    similarities = np.reshape(similarities, [-1, num_txt_per_img])
    idx_lst = np.reshape(idx_lst, [-1, num_txt_per_img])

    i2t_ranks, t2i_ranks, Aligned_lst = [], [], []
    for lab, sim, idx in zip(labels, similarities, idx_lst):
        inds = np.argsort(sim)[::-1]  # descending 4,3,2,1
        rank = num_txt_per_img
        for r, ind in enumerate(inds):
            if lab[ind] == 1:
                rank = r
                break
        Aligned_lst.append([idx[ind], rank])
        if args.i2t:
            i2t_ranks.append(rank)  # total len == len(dataset)
        elif args.t2i:
            t2i_ranks.append(rank)
        print('len of i2t_ranks, t2i_ranks:', len(i2t_ranks), len(t2i_ranks))
    return i2t_ranks, t2i_ranks, Aligned_lst

def compute_recall_precision(args, results, labels, idx_lst):
    labels = np.array(labels)
    similarities = np.array([results[i] for i in range(len(idx_lst))])

    num_txt_per_img = args.eval_len_size

    labels = np.reshape(labels, [-1, num_txt_per_img])
    similarities = np.reshape(similarities, [-1, num_txt_per_img])

    ranks = [1, 5, 10]
    recall, precision = [], []
    for k in ranks:
        r_lst, p_lst = [], []
        for lab, sim in zip(labels, similarities):
            sorted_label = []
            inds = np.argsort(sim)[::-1]  # descending 4,3,2,1
            for ind in inds:
                sorted_label.append(lab[ind])
            top = np.array(sorted_label[:k]).sum()
            bottom = np.array(sorted_label).sum()

            r = top / bottom
            p = top / k

            r_lst.append(r)
            p_lst.append(p)
        r_v = np.mean(np.array(r_lst))
        p_v = np.mean(np.array(p_lst))
        recall.append(r_v)
        precision.append(p_v)

    if args.i2t:
        results = {'i2t_recall': {"R@1": round(recall[0], 3), "R@5": round(recall[1], 3), "R@10": round(recall[2], 3)},
                   'i2t_precision': {"R@1": round(precision[0], 3), "R@5": round(precision[1], 3), "R@10": round(precision[2], 3)}}
    elif args.t2i:
        results = {'t2i_recall': {"R@1": round(recall[0], 3), "R@5": round(recall[1], 3), "R@10": round(recall[2], 3)},
                   't2i_precision': {"R@1": round(precision[0], 3), "R@5": round(precision[1], 3), "R@10": round(precision[2], 3)}}
    return results

def compute_mrr(ranks):
    ranks = np.array(ranks, dtype=float)
    ranks = ranks + 1
    print('ranks + 1:', ranks)
    mrr_score = np.mean(np.reciprocal(ranks))
    print('reciprocal_ranks:', np.reciprocal(ranks))
    print('mrr_score:', mrr_score)
    # mrr_score = np.mean(np.divide(1, ranks, out=np.zeros_like(ranks), where=ranks!=0))
    return mrr_score


def evaluate(args, test_results, test_labels, idx_lst):  # hits at n score, n = [1, 5, 10]
    i2t_ranks, t2i_ranks, Aligned_lst = compute_ranks(args, test_results, test_labels, idx_lst)
    recall_precision_results = compute_recall_precision(args, test_results, test_labels, idx_lst)
    rank = [1, 5, 10]
    eval_result = {}
    if args.i2t:
        i2t_accs = [sum([_ < r for _ in i2t_ranks]) / len(i2t_ranks) for r in rank]
        eval_result = {"i2t_retrieval": {"R@1": i2t_accs[0], "R@5": i2t_accs[1], "R@10": i2t_accs[2]}}
        mrr_score = compute_mrr(i2t_ranks)
    elif args.t2i:
        t2i_accs = [sum([_ < r for _ in t2i_ranks]) / len(t2i_ranks) for r in rank]
        eval_result["t2i_retrieval"] = {"R@1": t2i_accs[0], "R@5": t2i_accs[1], "R@10": t2i_accs[2]}
        mrr_score = compute_mrr(t2i_ranks)
    return eval_result, Aligned_lst, mrr_score, recall_precision_results



def train(args, train_dataset, val_dataset, model, bert, tokenizer, dset):
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    global_step, global_loss, global_acc = 0, 0.0, 0.0
    best_score = 0

    for epoch in range(int(args.epochs)):
        train_losses = []
        train_acc = []
        train_data_iter = tqdm(enumerate(train_dataset),
                               desc=f'EP_:{epoch}',
                               total=len(train_dataset),
                               bar_format='{l_bar}{r_bar}')
        for step, (_, batch) in train_data_iter:
            model.train()
            if args.CXRBERT:
                cls_tok = torch.cat((batch[0], batch[7]), dim=0).to(args.device)
                input_txt = torch.cat((batch[1], batch[8]), dim=0).to(args.device)
                attn_mask = torch.cat((batch[2], batch[9]), dim=0).to(args.device)
                input_img = torch.cat((batch[3], batch[10]), dim=0).to(args.device)
                segment = torch.cat((batch[4], batch[11]), dim=0).to(args.device)
                sep_tok = torch.cat((batch[5], batch[12]), dim=0).to(args.device)
                labels = torch.cat((batch[6], batch[13]), dim=0).to(args.device)
                logits = model(cls_tok, input_txt, attn_mask, segment, input_img, sep_tok)
            else:
                input_txt = torch.cat((batch[0], batch[5]), dim=0)
                # cls_tok = torch.cat((batch[1], batch[6]), dim=0)
                # segment = torch.cat((batch[2], batch[7]), dim=0)
                input_img = torch.cat((batch[3], batch[8]), dim=0)
                labels = torch.cat((batch[4], batch[9]), dim=0).to(args.device)

                patches = rearrange(input_img, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=16, s2=16)
                # cls_embed = bert.embeddings(cls_tok)
                txt_embed = bert.embeddings(input_txt)
                # print("\n")
                # print("patches", patches.shape)
                # print("txt_embed", txt_embed.shape)

                catput = torch.cat((txt_embed, patches), dim=1)
                # print("catput", catput.shape)

                # catput = torch.cat(cls_tok, input_txt, segment, input_img)
                """input_txt torch.Size([20, 130])
                segment torch.Size([140, 130])
                cls_tok torch.Size([140, 1])
                input_img torch.Size([140, 3, 512, 512])
                labels torch.Size([140])"""
                # input("STOP!!")

                logits = model(catput.to(args.device)).to(args.device)

            loss = criterion(logits.view(-1, 2), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logits = torch.max(logits, 1)[1].data  # argmax
            scores = logits == labels
            batch_score = scores.sum()
            batch_acc = batch_score.item() / (args.batch_size * 2)

            global_loss += loss.item()
            global_acc += batch_acc

            train_losses.append(loss.item())
            train_acc.append(batch_acc)

            print(f'Epoch : {epoch}, Step : {step}/{len(train_dataset)}, '
                  f'loss : {round(loss.item(), 3)}({round(np.mean(train_losses), 3)}), '
                  f'score : {round(batch_acc, 3)}({round(np.mean(train_acc), 3)})')

        wandb.log({
            "avg_loss": np.mean(train_losses),
            "retrieval_acc": np.mean(train_acc),
        }, step=epoch)

        save_path_per_ep = os.path.join(args.output_path, str(epoch))
        if not os.path.exists(save_path_per_ep):
            os.mkdir(save_path_per_ep)
            os.chmod(save_path_per_ep, 0o777)

        if args.n_gpu > 1:
            model.module.save_pretrained(save_path_per_ep)
            print(f'Multi_EP: {epoch} Model saved on {save_path_per_ep}')
        else:
            model.save_pretrained(save_path_per_ep)
            print(f'Single_EP: {epoch} Model saved on {save_path_per_ep}')

        # Evaluate during training
        if args.eval_during_training:  # and epoch > 4:
            test_result, test_label, test_losses, idx_lst = test(args, model, val_dataset)
            eval_result, Aligned_lst, mrr_score, recall_precision_results = evaluate(args, test_result, test_label, idx_lst)

            file_data = OrderedDict()
            result_path = os.path.join(args.output_path, 'rank_result_at_eval.json')
            for result in Aligned_lst:
                idx = result[0]
                rank = result[1]
                data = dset.data[idx]
                with open(result_path, 'a', encoding='utf-8') as make_file:
                    file_data["Rank"] = rank
                    file_data["Result"] = data
                    json.dump(file_data, make_file, ensure_ascii=False)
                    make_file.write('\n')

            if args.i2t:
                assert not args.t2i
                rank_accs = eval_result['i2t_retrieval']
                precision = recall_precision_results['i2t_precision']
                recall = recall_precision_results['i2t_recall']
            elif args.t2i:
                assert not args.i2t
                rank_accs = eval_result['t2i_retrieval']
                precision = recall_precision_results['t2i_precision']
                recall = recall_precision_results['t2i_recall']

            if rank_accs['R@1'] > best_score:
                best_score = rank_accs['R@1']

            H1, H5, H10 = rank_accs['R@1'], rank_accs['R@5'], rank_accs['R@10']
            R1, R5, R10 = recall['R@1'], recall['R@5'], recall['R@10']
            P1, P5, P10 = precision['R@1'], precision['R@5'], precision['R@10']

            print(f'Eval during Training, Epoch:{epoch}, MRR_score:{mrr_score}'
                  f'Hit@1:{H1}, Hit@5:{H5}, Hit@10:{H10}, best_Hit1:{best_score},'
                  f'Recall@1:{R1}, Recall@5:{R5}, Recall@10:{R10},'
                  f'Precision@1:{P1}, Precision@1:{P5}, Precision@1:{P10}')
            wandb.log({
                "Hit@1": H1, "Hit@5": H5, "Hit@10": H10,
                "Recall@1": R1, "Recall@5": R5, "Recall@10": R10,
                "Precision@1": P1, "Precision@5": P5, "Precision@10": P10,
                "best_Hit1": best_score, "test_loss": np.mean(test_losses), "mrr_score": mrr_score,
            }, step=epoch)


def test(args, model, eval_dataset):
    model.eval()
    labels = []
    results_lst = []
    idx_lst = []
    softmax = nn.Softmax(dim=1)
    criterion = nn.CrossEntropyLoss()
    eval_losses = []

    eval_data_iter = tqdm(enumerate(eval_dataset),
                          total=len(eval_dataset),
                          bar_format='{l_bar}{r_bar}')

    for idx, batch in eval_data_iter:
        with torch.no_grad():
            if args.CXRBERT:
                cls_tok = batch[0].to(args.device)
                input_txt = batch[1].to(args.device)
                attn_mask = batch[2].to(args.device)
                input_img = batch[3].to(args.device)
                segment = batch[4].to(args.device)
                sep_tok = batch[5].to(args.device)

                label = batch[6].tolist()
                idx = batch[7].tolist()

                logits = model(cls_tok, input_txt, attn_mask, segment, input_img, sep_tok)
            else:
                input_txt = batch[0].to(args.device)
                attn_mask = batch[1].to(args.device)
                input_img = batch[2].to(args.device)
                segment = batch[3].to(args.device)

                label = batch[4].tolist()
                idx = batch[5].tolist()

                # logits = model(input_txt, attn_mask, input_img, segment)
                logits = model(cls_tok, input_txt, segment, input_img)

            labels.extend(label)
            idx_lst.extend(idx)

            eval_loss = criterion(logits, torch.tensor(label).to(args.device))
            eval_losses.append(eval_loss.item())

            probs = softmax(logits)
            result = probs[:, 1]  # the confidence to be a matched pair (1)

            result = [_.to(torch.device("cpu")) for _ in result]
            results_lst.extend(result)
    return results_lst, labels, eval_losses, idx_lst  # results, labels

def get_transforms(args):
    norm = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    if args.img_size == 224:
        return transforms.Compose(
            [transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(*norm)])
    elif args.img_size == 512:
        return transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(*norm)])  # referred from ChexNet])


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout))


def MLPMixer(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor = 4, dropout = 0.):
    # assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
    # num_patches = (image_size // patch_size) ** 2
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(image_size, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes))

class ImageTextMatching(nn.Module):
    """
    2-class classification model : Aligned, Not aligned
    """
    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, 2)

    def forward(self, x):
        return self.linear(x)

class CXR_mixer(nn.Module):
    def __init__(self, config, args):
        super(CXR_mixer, self).__init__()
        self.cxrmixer = MLPMixer(image_size = 326, channels = 3, patch_size = 16, dim = 768, depth = 12, num_classes = 2)

    def forward(self, catput):
        pred = self.cxrmixer(catput)
        # print("pred", pred)
        return pred

def main(args):
    wandb.init(config=args, project='fianl_project', entity='mimic-cxr')
    cuda_condition = torch.cuda.is_available() and args.with_cuda
    args.device = torch.device("cuda" if cuda_condition else "cpu")
    args.n_gpu = torch.cuda.device_count()
    print(f'Device: {args.device}, n_gpu: {args.n_gpu}')

    config = BertConfig.from_pretrained("bert-base-uncased")
    transforms = get_transforms(args)
    bert = BertModel(config)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True).tokenize

    if args.CXRBERT:
        model = CXR_mixer(config, args).to(args.device)
    else:
        if args.do_train:
            model = CXR_mixer(config, args).to(args.device)
        elif args.do_test:
            config = AutoConfig.from_pretrained(args.load_pretrained_model)
            model_state_dict = torch.load(os.path.join(args.load_pretrained_model, 'pytorch_model.bin'))
            model = CXR_mixer.from_pretrained(args.load_pretrained_model, state_dict=model_state_dict, config=config, args=args).to(args.device)

    wandb.watch(model)
    if args.with_cuda and args.n_gpu > 1:
        model = nn.DataParallel(model, device_ids=args.cuda_devices)

    if args.do_train:
        print("Load transforms", transforms)
        train_dataset = CXR_Retrieval_Dataset(args.train_dataset, tokenizer, transforms, args, is_train=True)

        if args.label_conditioned:
            val_dataset = CXR_Retrieval_Dataset(args.label_conditioned_valid_dataset, tokenizer, transforms, args, is_train=False)
        else:
            val_dataset = CXR_Retrieval_Dataset(args.studyID_valid_dataset, tokenizer, transforms, args, is_train=False)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        eval_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

        train(args, train_dataloader, eval_dataloader, model, bert, tokenizer, val_dataset)


    if args.do_test:
        if args.label_conditioned:
            print("Load Test dataset", args.label_conditioned_test_dataset)
            test_dataset = CXR_Retrieval_Dataset(args.label_conditioned_test_dataset, tokenizer, transforms, args, is_train=False)
        else:
            print("Load Test dataset", args.studyID_test_dataset)
            test_dataset = CXR_Retrieval_Dataset(args.studyID_test_dataset, tokenizer, transforms, args, is_train=False)
        eval_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
        best_score = 0

        test_result, test_label, test_losses, idx_lst = test(args, model, eval_dataloader)
        eval_result, Aligned_lst, mrr_score, recall_precision_results = evaluate(args, test_result, test_label, idx_lst)

        file_data = OrderedDict()
        result_path = os.path.join(args.output_path, 'rank_result_at_eval.json')
        for result in Aligned_lst:
            idx = result[0]
            rank = result[1]
            data = test_dataset.data[idx]
            with open(result_path, 'a', encoding='utf-8') as make_file:
                file_data["Rank"] = rank
                file_data["Result"] = data
                json.dump(file_data, make_file, ensure_ascii=False)
                make_file.write('\n')

        file_data = OrderedDict()
        result_path = os.path.join(args.output_path, 'rank_result_at_eval.json')
        for result in Aligned_lst:
            idx = result[0]
            rank = result[1]
            data = test_dataset.data[idx]
            with open(result_path, 'a', encoding='utf-8') as make_file:
                file_data["Rank"] = rank
                file_data["Result"] = data
                json.dump(file_data, make_file, ensure_ascii=False)
                make_file.write('\n')

        if args.i2t:
            assert not args.t2i
            rank_accs = eval_result['i2t_retrieval']
            precision = recall_precision_results['i2t_precision']
            recall = recall_precision_results['i2t_recall']
        elif args.t2i:
            assert not args.i2t
            rank_accs = eval_result['t2i_retrieval']
            precision = recall_precision_results['t2i_precision']
            recall = recall_precision_results['t2i_recall']

        if rank_accs['R@1'] > best_score:
            best_score = rank_accs['R@1']

        H1, H5, H10 = rank_accs['R@1'], rank_accs['R@5'], rank_accs['R@10']
        R1, R5, R10 = recall['R@1'], recall['R@5'], recall['R@10']
        P1, P5, P10 = precision['R@1'], precision['R@5'], precision['R@10']

        print(f'At TEST, mrr_score:{mrr_score}'
              f'Hit@1:{H1}, Hit@5:{H5}, Hit@10:{H10}, best_Hit1:{best_score},'
              f'Recall@1:{R1}, Recall@5:{R5}, Recall@10:{R10},'
              f'Precision@1:{P1}, Precision@5:{P5}, Precision@10:{P10}')

        wandb.log({
            "F_Hit@1": H1, "F_Hit@5": H5, "F_Hit@10": H10,
            "F_Recall@1": R1, "F_Recall@5": R5, "F_Recall@10": R10,
            "F_Precision@1": P1, "F_Precision@5": P5, "F_Precision@10": P10,
            "F_best_Hit1": best_score, "F_test_loss": np.mean(test_losses), "F_mrr_score": mrr_score,
        },)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # TODO: for Retrieval
    parser.add_argument("--t2i", type=bool, default=True, help="Text-to-Image Retrieval")
    parser.add_argument("--i2t", type=bool, default=False, help="Image-to-Text Retrieval")
    # TODO: !!!!!!!!!!! MIMIC(val, test) or OPENI(val, test)
    parser.add_argument("--eval_len_size", type=int, default=354, choices=[759, 1536, 710, 354],
                        help="example size per idx_matching_example")  # 759

    # TODO: MODEL, CXRBERT or CNN_BERT
    parser.add_argument("--CXRBERT", type=bool, default=False, help="CXRBERT(T), CNN_BERT(F)")
    parser.add_argument("--do_train", type=bool, default=True, help="Train & Evaluate")
    parser.add_argument("--do_test", type=bool, default=False, help="Test")

    # eval_during_training
    # must be deleted! after validation dataset
    # TODO: only MIMIC, PAR, set to True if not set to False
    parser.add_argument("--eval_during_training", type=bool, default=True, help="eval_druing_training")

    # TODO: label_conditioned or just study_id matching !
    # TODO: Choose dataset, mimic or openI
    parser.add_argument("--MIMIC_dset", type=bool, default=True,
                        help="using mimic-cxr dataset(T), using openi dataset (F)")
    parser.add_argument("--label_conditioned", type=bool, default=True,
                        help="label_conditioned(T), Study_ID matching (F)")

    # TODO: trainset, mimic or openi
    parser.add_argument("--train_dataset", type=str,
                        default='/home/data_storage/mimic-cxr/dataset/image_preprocessing/Train.jsonl',
                        choices=['/home/data_storage/mimic-cxr/dataset/image_preprocessing/Train.jsonl',
                                 '/home/data_storage/mimic-cxr/dataset/open_i/Train_openi.jsonl'],
                        help="train dataset for training")

    parser.add_argument("--label_conditioned_valid_dataset", type=str,
                        default='/home/data_storage/mimic-cxr/dataset/dataset/retrieval/dset/T2I_Label_Valid.jsonl',
                        help='label conditioned valid dataset for evaluating train set',
                        choices=['/home/data_storage/mimic-cxr/dataset/dataset/retrieval/dset/T2I_Label_Valid.jsonl',
                                 '/home/data_storage/mimic-cxr/dataset/dataset/retrieval/dset/I2T_Label_Valid.jsonl',
                                 '/home/data_storage/mimic-cxr/dataset/dataset/retrieval/dset/openI/T2I_Label_Valid_openi.jsonl',
                                 '/home/data_storage/mimic-cxr/dataset/dataset/retrieval/dset/openI/I2T_Label_Valid_openi.jsonl',])

    parser.add_argument("--label_conditioned_test_dataset", type=str,
                        default='/home/data_storage/mimic-cxr/dataset/dataset/retrieval/dset/T2I_Label_Test.jsonl',
                        help='label conditioned test dataset for evaluating the model',
                        choices=['/home/data_storage/mimic-cxr/dataset/dataset/retrieval/dset/T2I_Label_Test.jsonl',
                                 '/home/data_storage/mimic-cxr/dataset/dataset/retrieval/dset/I2T_Label_Test.jsonl',
                                 '/home/data_storage/mimic-cxr/dataset/retrieval/openi/T2I_Label_Test_openi.jsonl',
                                 '/home/data_storage/mimic-cxr/dataset/retrieval/openi/I2T_Label_Test_openi.jsonl',])

    #######################################
    parser.add_argument("--studyID_valid_dataset", type=str,
                        default='/home/data_storage/mimic-cxr/dataset/retrieval/openi/T2I_ID_Test_openi.jsonl',
                        help='only study id valid dataset for evaluating train set',
                        choices=['/home/data_storage/mimic-cxr/dataset/dataset/retrieval/dset/T2I_ID_Valid.jsonl',
                                 '/home/data_storage/mimic-cxr/dataset/dataset/retrieval/dset/I2T_ID_Valid.jsonl',
                                 '/home/mimic-cxr/dataset/retrieval/openI/T2I_ID_Valid_openi.jsonl',
                                 '/home/mimic-cxr/dataset/retrieval/openI/I2T_ID_Valid_openi.jsonl',])

    parser.add_argument("--studyID_test_dataset", type=str,
                        default='/home/data_storage/mimic-cxr/dataset/retrieval/openi/T2I_ID_Test_openi.jsonl',
                        help='only study id test dataset for evaluating the model',
                        choices=['/home/data_storage/mimic-cxr/dataset/dataset/retrieval/dset/T2I_ID_Test.jsonl',
                                 '/home/data_storage/mimic-cxr/dataset/dataset/retrieval/dset/I2T_ID_Test.jsonl',
                                 '/home/data_storage/mimic-cxr/dataset/retrieval/openi/T2I_ID_Test_openi.jsonl',
                                 '/home/data_storage/mimic-cxr/dataset/retrieval/openi/I2T_ID_Test_openi.jsonl',])

    output_path = 'output/' + str(datetime.now())
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        os.chmod(output_path, 0o777)

    parser.add_argument("--output_path", type=str, default=output_path, help="ex)path/to/save/model")
    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: True or False")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")

    parser.add_argument("--epochs", type=int, default=50, help='number of epochs')
    parser.add_argument("--batch_size", type=int, default=128, help="number of batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="dataloader worker size")

    # TODO: load pre-trained model or not
    parser.add_argument("--hidden_size", type=int, default=768, choices=[768, 512, 128])
    parser.add_argument("--embedding_size", type=int, default=768, choices=[768, 512, 128])

    # When args.CXRBERT: TRUE
    # Load pre-trained model -> weight_load(T), load_pretrained_model: change path, bert_model: set to same size model
    # From scratch -> weight_load(F), bert_model: set to bert-base-scratch(in cxrbert_origin.py, CXRBertEncoder)
    parser.add_argument("--weight_load", type=bool, default=True, help='load_pretrained_model(T), scratch(F)')

    # do_train -> load_pretrained_model: Pre-trained CXR-BERT
    # do_test -> load_pretrained_model: saved CXRBertForRetrieval model path,
    parser.add_argument("--load_pretrained_model", type=str,
                        default='/home/data_storage/mimic-cxr/models/retrieval_openi/openi_base_sc_par_label',

                        choices=['/home/mimic-cxr/model_hglee/model_scp/retrieval/CXRBERT_T2I_full_label_base_sc_par_180,253/4',
                                 '/home/mimic-cxr/model_hglee/model_scp/retrieval/CXRBERT_I2T_full_label_base_sc_par_180,253/4',
                                 '/home/mimic-cxr/model_hglee/model_scp/retrieval/CXRBERT_T2I_full_label_base_sc_bi_180,253/2',
                                 '/home/mimic-cxr/model_hglee/model_scp/retrieval/CNN_BERT_base_sc_I2T_full_label/9',  # Testset
                                 '/home/mimic-cxr/model_hglee/model_scp/retrieval/CNN_BERT_base_sc_T2I_full_label/9',  # Testset
                                 '/home/mimic-cxr/model_hglee/model_scp2/Base_sc_180_253_par',
                                 '/home/mimic-cxr/model_hglee/model_scp2/Base_sc_180_253_s2s',
                                 '/home/mimic-cxr/model_hglee/model_scp2/Base_sc_180_253_bi',

                                 '/home/data_storage/mimic-cxr/models/Base_sc_180_253_bi',
                                 '/home/data_storage/mimic-cxr/models/Base_sc_180_253_s2s',
                                 '/home/data_storage/mimic-cxr/models/Base_sc_180_253_bi_s2s/49',])  # BERT-small

    # TODO: Model size, both CXRBERT and CNN_BERT
    parser.add_argument("--bert_model", type=str, default="bert-base-scratch",
                        choices=["albert-base-v2",
                                 "bert-base-uncased",
                                 "google/bert_uncased_L-4_H-512_A-8",  # BERT-Small
                                 "google/bert_uncased_L-2_H-128_A-2",  # BERT-Tiny
                                 "emilyalsentzer/Bio_ClinicalBERT",  # Clinical-BERT
                                 "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",  # BlueBERT
                                 "bert-small-scratch",  # BERT-Small-scratch
                                 "bert-base-scratch", ])

    parser.add_argument("--vocab_size", type=int, default=30522, choices=[30522, 30000, 28996])  # 28996 clinical bert

    parser.add_argument("--img_postion", default=True, help='img_postion use!')
    parser.add_argument("--seq_len", type=int, default=128, help="maximum sequence len", choices=[253, 460])  # 253
    parser.add_argument("--max_seq_len", type=int, default=512, help="total sequence len")

    parser.add_argument("--img_hidden_sz", type=int, default=2048)
    parser.add_argument("--img_encoder", type=str, default='full-fiber',
                        choices=['random-pixel', 'full-fiber', 'ViT'])
    # TODO: MIMIC OR OPENI, 3 or 1 channel
    parser.add_argument("--img_channel", type=int, default=1, choices=[1, 3])
    parser.add_argument("--num_image_embeds", type=int, default=256, choices=[36, 49, 256])
    parser.add_argument("--img_size", type=int, default=224, choices=[224, 512])  # TODO: change helper.py, resize(224)
    parser.add_argument("--img_embed_pool_type", type=str, default="max", choices=["max", "avg"])

    # -------------------------------------------------------------------------------------------
    # TODO: ...!
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)  # loss, optimizer.step() slowly
    parser.add_argument("--warmup", type=float, default=0.1)  # optimizer = BertAdam(warmup=args.warmup)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--dropout_prob", type=float, default=0.1)

    parser.add_argument("--beta1", type=float, default=0.9, help="adams first beta value")
    parser.add_argument("--beta2", type=float, default=0.999, help="adams first beta value")
    parser.add_argument("--eps", type=float, default=1e-6, help="adams epsilon")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight_decay of AdamW")  # 0.01 , AdamW

    args = parser.parse_args()

    main(args)

