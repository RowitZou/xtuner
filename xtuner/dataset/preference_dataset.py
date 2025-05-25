# Copyright (c) OpenMMLab. All rights reserved.
import copy
import json
import os
from datetime import timedelta
from functools import partial
from multiprocessing import Process, Queue
from typing import Callable, Dict, List, Optional

import numpy as np
import torch.distributed as dist
import tqdm
from datasets import Dataset as HFDataset
from datasets import concatenate_datasets, load_dataset, DatasetDict
from mmengine.config import Config, ConfigDict
from mmengine.logging import print_log
from mmengine.utils.misc import get_object_from_string
from torch.utils.data import Dataset, IterableDataset
from transformers import AutoTokenizer

from xtuner.registry import BUILDER, MAP_FUNC

from .huggingface import build_origin_dataset


def _worker(
    tokenize_fun: Callable,
    data_queue: Queue,
    out_queue: Queue,
):
    while True:
        data_chunk = data_queue.get()

        if data_chunk is None:
            out_queue.put(None)
            break
        chunk_results = []
        for idx, data in data_chunk:
            chunk_results.append([idx, tokenize_fun(data)])
        out_queue.put(chunk_results)


def _chunk_data_to_queue(data_queue: Queue, data: List[Dict], chunk_size: int, nproc):
    data_iter = iter(data)
    chunk_data = []
    while True:
        try:
            item = next(data_iter)
        except StopIteration:
            break
        chunk_data.append(item)
        if len(chunk_data) == chunk_size:
            data_queue.put(chunk_data)
            chunk_data = []
    if chunk_data:
        data_queue.put(chunk_data)

    for _ in range(nproc):
        data_queue.put(None)


def _multi_progress(tokenize_fun_p, dataset, nproc, task_num, chunksize, description):
    processes = []
    data_queue = Queue()
    output_queue = Queue()
    bar = tqdm.tqdm(total=task_num, desc=description)
    # task_id = bar.add_task(total=task_num, description=description)
    dataset = enumerate(dataset)
    _chunk_data_to_queue(data_queue, dataset, chunksize, nproc)
    for _ in range(nproc):
        process = Process(
            target=_worker, args=(tokenize_fun_p, data_queue, output_queue)
        )
        process.start()
        processes.append(process)

    results = []
    finished_process = 0
    while finished_process < nproc:
        chunk_results = output_queue.get()
        if chunk_results is None:
            finished_process += 1
            continue
        results.extend(chunk_results)
        bar.update(len(chunk_results))
        bar.refresh()
    results = map(lambda x: x[1], sorted(results, key=lambda x: x[0]))
    return results


def load_jsonl_dataset(data_files=None, data_dir=None, suffix=None):
    assert (data_files is not None) != (data_dir is not None)
    if data_dir is not None:
        data_files = os.listdir(data_dir)
        data_files = [os.path.join(data_dir, fn) for fn in data_files]
        if suffix is not None:
            data_files = [fp for fp in data_files if fp.endswith(suffix)]
    elif isinstance(data_files, str):
        data_files = [data_files]

    dataset_list = []
    for fp in data_files:
        with open(fp, encoding="utf-8") as file:
            data = [json.loads(line) for line in file]
        ds = HFDataset.from_list(data)
        dataset_list.append(ds)
    dataset = concatenate_datasets(dataset_list)
    return dataset


def tokenize(
    pair: str,
    tokenizer: AutoTokenizer,
    max_length: int,
    is_reward: bool = False,
    reward_token_id: int = -1,
):
    prompt = tokenizer.apply_chat_template(
        pair["prompt"], tokenize=False, add_generation_prompt=True
    )
    chosen = tokenizer.apply_chat_template(
        pair["prompt"] + pair["chosen"], tokenize=False, add_generation_prompt=False
    )
    rejected = tokenizer.apply_chat_template(
        pair["prompt"] + pair["rejected"], tokenize=False, add_generation_prompt=False
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    chosen_ids = tokenizer.encode(chosen, add_special_tokens=False)
    rejected_ids = tokenizer.encode(rejected, add_special_tokens=False)

    if len(chosen_ids) > max_length:
        chosen_ids = chosen_ids[:max_length]
    if len(rejected_ids) > max_length:
        rejected_ids = rejected_ids[:max_length]

    if is_reward:
        # reward label
        chosen_ids = chosen_ids + [reward_token_id]
        rejected_ids = rejected_ids + [reward_token_id]
        chosen_labels = [-100] * len(chosen_ids[:-1]) + [0]
        rejected_labels = [-100] * len(rejected_ids[:-1]) + [1]
    else:
        # dpo label
        prompt_len = min(len(prompt_ids), max_length)
        chosen_labels = [-100] * prompt_len + copy.deepcopy(chosen_ids[prompt_len:])
        rejected_labels = [-100] * prompt_len + copy.deepcopy(rejected_ids[prompt_len:])

    return {
        "chosen_ids": chosen_ids,
        "rejected_ids": rejected_ids,
        "chosen_labels": chosen_labels,
        "rejected_labels": rejected_labels,
    }


def tokenize_rmp(
    pair: str,
    idx: int,
    tokenizer: AutoTokenizer,
    max_length: int,
    max_response_length: int,
    is_reward: bool = False,
    reward_token_id: int = -1
):

    max_length = max_length - 4  # for one RM token and two seperator tokens and one reward token

    prompt_ddm = "\n".join([e["content"] for e in (pair["prompt"])])
    reference_ddm = "\n".join([e["content"] for e in (pair["reference"])])
    chosen_ddm = "\n".join([e["content"] for e in (pair["chosen"])])
    rejected_ddm = "\n".join([e["content"] for e in (pair["rejected"])])
    wrapper = pair['wrapper']

    _prompt_ids = tokenizer.encode(prompt_ddm, add_special_tokens=True)
    _reference_ids = tokenizer.encode(reference_ddm, add_special_tokens=True)
    _chosen_ids = tokenizer.encode(chosen_ddm, add_special_tokens=True)
    _rejected_ids = tokenizer.encode(rejected_ddm, add_special_tokens=True)

    if len(_reference_ids) > max_response_length:
        print_log(
            f"sequence length {len(_reference_ids)} is "
            f"larger than max_response_length {max_response_length}",
            logger="current",
        )
        _reference_ids = _reference_ids[:max_response_length]
    if len(_chosen_ids) > max_response_length:
        print_log(
            f"sequence length {len(_chosen_ids)} is "
            f"larger than max_response_length {max_response_length}",
            logger="current",
        )
        _chosen_ids = _chosen_ids[:max_response_length]
    if len(_rejected_ids) > max_response_length:
        print_log(
            f"sequence length {len(_rejected_ids)} is "
            f"larger than max_response_length {max_response_length}",
            logger="current",
        )
        _rejected_ids = _rejected_ids[:max_response_length]

    max_prompt_length = min((max_length - len(_reference_ids) - len(_chosen_ids)) // 2,
                            (max_length - len(_reference_ids) - len(_rejected_ids)) // 2)

    if len(_prompt_ids) > max_prompt_length:
        print_log(
            f"sequence length {len(_prompt_ids)} is "
            f"larger than max_prompt_length {max_prompt_length}",
            logger="current",
        )
        # _prompt_ids = _prompt_ids[:max_prompt_length]
        # 这里的逻辑是为了保证 prompt 的长度不超过 max_prompt_length
        # 但是如果直接截断，可能会导致语义不完整，所以这里选择保留最后 max_prompt_length 个 token
        # 这样可以保证语义的完整性，同时也能满足长度的要求
        _prompt_ids = _prompt_ids[-max_prompt_length:]

    _prompt = tokenizer.decode(_prompt_ids, skip_special_tokens=True)
    _reference = tokenizer.decode(_reference_ids, skip_special_tokens=True)
    _chosen = tokenizer.decode(_chosen_ids, skip_special_tokens=True)
    _rejected = tokenizer.decode(_rejected_ids, skip_special_tokens=True)

    # Fit the template of RMP
    _reference_cat = _prompt + _reference if wrapper == "pretrain" or _reference == "" else _prompt + "\n" + _reference
    _chosen_cat = _prompt + _chosen if wrapper == "pretrain" or _chosen == "" else _prompt + "\n" + _chosen
    _rejected_cat = _prompt + _rejected if wrapper == "pretrain" or _rejected == "" else _prompt + "\n" + _rejected

    chosen = _reference_cat + "<|reward|>" + _chosen_cat
    rejected = _reference_cat + "<|reward|>" + _rejected_cat

    chosen_ids = tokenizer.encode(chosen, add_special_tokens=True)
    rejected_ids = tokenizer.encode(rejected, add_special_tokens=True)

    if is_reward:
        # reward label
        chosen_ids = chosen_ids + [reward_token_id]
        rejected_ids = rejected_ids + [reward_token_id]
        chosen_labels = [-100] * len(chosen_ids[:-1]) + [0]
        rejected_labels = [-100] * len(rejected_ids[:-1]) + [1]
    else:
        raise NotImplementedError

    return {
        'idx': idx,
        'chosen_ids': chosen_ids,
        'rejected_ids': rejected_ids,
        'chosen_labels': chosen_labels,
        'rejected_labels': rejected_labels,
    }


class PreferenceDataset(Dataset):
    def __init__(
        self,
        dataset: HFDataset,
        tokenizer: AutoTokenizer,
        max_length: int,
        max_response_length: int,
        is_dpo: bool = True,
        is_reward: bool = False,
        reward_token_id: int = -1,
        num_proc: int = 32,
    ) -> None:
        self.max_length = max_length
        self.max_response_length = max_response_length
        assert is_dpo != is_reward, "Only one of is_dpo and is_reward can be True"
        if is_reward:
            assert (
                reward_token_id != -1
            ), "reward_token_id should be set if is_reward is True"

        self.is_dpo = is_dpo
        self.is_reward = is_reward
        self.reward_token_id = reward_token_id
        self.tokenized_pairs = []

        for tokenized_pair in _multi_progress(
            partial(
                tokenize_rmp,
                idx=-1,
                tokenizer=tokenizer,
                max_length=max_length,
                max_response_length=max_response_length,
                is_reward=is_reward,
                reward_token_id=reward_token_id,
            ),
            dataset,
            nproc=num_proc,
            task_num=len(dataset),
            chunksize=num_proc,
            description="Tokenizing dataset",
        ):
            self.tokenized_pairs.append(tokenized_pair)

    def __len__(self):
        return len(self.tokenized_pairs)

    def __getitem__(self, idx):
        return self.tokenized_pairs[idx]


class PretrainPreferenceDatasetStream(IterableDataset):
    def __init__(
        self,
        root_path: str,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        work_dir_name: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.work_dir_name = os.path.join("/cpfs01/shared/alillm_hs/zouyicheng/xtuner/work_dirs", work_dir_name, "last_ckpt_data_idxes")
        self.rank = 0 if rank is None else rank
        self.world_size = 1 if world_size is None else world_size

        self.dataset_dict = DatasetDict({
            subdir: load_dataset(
                os.path.join(root_path, subdir),
                streaming=True
            )["train"]
            for subdir in os.listdir(root_path)
            if os.path.isdir(os.path.join(root_path, subdir))
        })

        self.dataset_shard_num_map = {'0_1~100_chat_3~4': 10200000,
                                      '10_1~100_chat_23~24': 10200000,
                                      '11_1~100_chat_25~26': 10200000,
                                      '12_1~100_chat_27~28': 10200000,
                                      '13_1~100_chat_29~30': 10200000,
                                      '14_1~100_chat_31~32': 10200000,
                                      '15_1~100_chat_33~34': 10200000,
                                      '16_1~100_chat_35~36': 10200000,
                                      '17_1~100_chat_37~38': 10200000,
                                      '18_1~100_chat_39~40': 10200000,
                                      '19_1~100_chat_41~42': 10200000,
                                      '1_1~100_chat_5~6': 10200000,
                                      '20_1~100_chat_43~44': 10200000,
                                      '21_1~100_chat_45~46': 10200000,
                                      '22_1~100_chat_47~48': 10200000,
                                      '23_1~100_chat_49~50': 10200000,
                                      '24_1~100_chat_51~52': 10200000,
                                      '25_1~100_chat_53~54': 10200000,
                                      '26_1~100_chat_55~56': 10200000,
                                      '27_1~100_chat_57~58': 10200000,
                                      '28_1~100_chat_59~60': 10200000,
                                      '29_1~100_chat_61~62': 10200000,
                                      '2_1~100_chat_7~8': 10200000,
                                      '30_1~100_chat_63~64': 10200000,
                                      '31_1~100_chat_65~66': 10200000,
                                      '32_1~100_chat_67~68': 10200000,
                                      '33_1~100_chat_69~70': 10200000,
                                      '34_1~100_chat_71~72': 10200000,
                                      '35_1~100_chat_73~74': 10200000,
                                      '36_1~100_chat_75~76': 10199380,
                                      '37_1~100_chat_77~78': 10200000,
                                      '38_1~100_chat_79~80': 10200000,
                                      '39_1~100_chat_81~82': 10200000,
                                      '3_1~100_chat_9~10': 10200000,
                                      '40_1~100_chat_83~84': 10200000,
                                      '41_1~100_chat_85~86': 10200000,
                                      '42_1~100_chat_87~88': 10200000,
                                      '43_1~100_chat_89~90': 10200000,
                                      '44_1~100_chat_91~92': 10200000,
                                      '45_1~100_chat_93~94': 10200000,
                                      '46_1~100_chat_95~96': 10200000,
                                      '47_1~100_chat_97~98': 10200000,
                                      '48_1~100_chat_99~100': 10190000,
                                      '4_1~100_chat_11~12': 10200000,
                                      '5_1~100_chat_13~14': 10200000,
                                      '6_1~100_chat_15~16': 10200000,
                                      '7_1~100_chat_17~18': 10200000,
                                      '8_1~100_chat_19~20': 10200000,
                                      '9_1~100_chat_21~22': 10200000,
                                      'p_1~99_chat_1~2': 9995996}

        last_data_resume_record = os.path.join(self.work_dir_name, str(self.rank))
        if os.path.exists(last_data_resume_record):
            with open(last_data_resume_record, "r") as f:
                last_batch_idxes = json.loads(f.readline())
            print(
                f"Rank {self.rank} resumes data from {last_data_resume_record}: {str(last_batch_idxes)}"
            )
            offset = self.world_size - 1 - self.rank
            self.last_batch_idx = last_batch_idxes[-1] + offset
        else:
            self.last_batch_idx = -1

        self.data_offset = self.last_batch_idx + 1

    def __iter__(self):

        for name, num in self.dataset_shard_num_map.items():
            if self.last_batch_idx + 1 >= num:
                self.last_batch_idx -= num
                print(f"Skipping {num} samples of file {name} in rank {self.rank}.")
                continue
            dataset = self.dataset_dict[name]
            for i, data in enumerate(dataset):
                if i <= self.last_batch_idx:
                    if i == self.last_batch_idx:
                        print(
                            f"Skipping data to sample {i + 1} for rank {self.rank}."
                        )
                        self.last_batch_idx = -1
                    elif i > 0 and i % 1000000 == 0:
                        print(
                            f"Enumerate file {name} and skipped {i} samples in rank {self.rank}."
                        )
                    continue
                yield data


class PreferenceDatasetStream(IterableDataset):
    def __init__(
        self,
        dataset,
        tokenizer: AutoTokenizer,
        max_length: int,
        max_response_length: int,
        is_dpo: bool = True,
        is_reward: bool = False,
        reward_token_id: int = -1,
        data_num: int = 0,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        use_varlen_attn: bool = False,
    ) -> None:

        super().__init__()
        self.max_length = max_length
        self.max_response_length = max_response_length
        assert is_dpo != is_reward, "Only one of is_dpo and is_reward can be True"
        if is_reward:
            assert (
                reward_token_id != -1
            ), "reward_token_id should be set if is_reward is True"

        self.is_dpo = is_dpo
        self.is_reward = is_reward
        self.reward_token_id = reward_token_id
        self.data_num = data_num
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.use_varlen_attn = use_varlen_attn

        # 如果没传 rank/world_size，默认 single-process 模式
        self.rank = 0 if rank is None else rank
        self.world_size = 1 if world_size is None else world_size

        if hasattr(self.dataset, "data_offset"):
            self.data_offset = self.dataset.data_offset
        else:
            self.data_offset = 0

    def __len__(self):
        return self.data_num // self.world_size

    def __iter__(self):

        for i, data in enumerate(self.dataset):
            if i == 0 and self.rank == 0:
                print_log(
                    f"Sampled data: {data.keys()}",
                    logger="current",
                )
                for k, v in data.items():
                    print_log(
                        f"{k}------- {v}",
                        logger="current",
                    )

            if i % self.world_size == self.rank:
                yield tokenize_rmp(
                    data,
                    self.data_offset + i,
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    max_response_length=self.max_response_length,
                    is_reward=self.is_reward,
                    reward_token_id=self.reward_token_id,
                )


class PackedDatasetWrapper(Dataset):
    def __init__(
        self, dataset, max_packed_length=16384, shuffle_before_pack=True
    ) -> None:
        super().__init__()
        self.max_packed_length = max_packed_length
        self.lengths = []
        self.data = []

        indices = np.arange(len(dataset))
        if shuffle_before_pack:
            np.random.shuffle(indices)

        data_bin = []
        bin_seq_len = 0
        removed = 0
        for idx in indices:
            data = dataset[int(idx)]
            cur_len = len(data["chosen_ids"]) + len(data["rejected_ids"])
            if cur_len > max_packed_length:
                print_log(
                    f"sequence length {cur_len} is "
                    f"larger than max_packed_length {max_packed_length}",
                    logger="current",
                )
                removed += 1
                continue
            if (bin_seq_len + cur_len) > max_packed_length and len(data_bin) > 0:
                self.data.append(data_bin)
                self.lengths.append(bin_seq_len)
                data_bin = []
                bin_seq_len = 0
            data_bin.append(data)
            bin_seq_len += cur_len

        if len(data_bin) > 0:
            self.data.append(data_bin)
            self.lengths.append(bin_seq_len)
        if removed > 0:
            print_log(
                f"removed {removed} samples because "
                f"of length larger than {max_packed_length}",
                logger="current",
            )
        print_log(
            f"The batch numbers of dataset is changed "
            f"from {len(dataset)} to {len(self)} after"
            " using var len attention.",
            logger="current",
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        pairs = self.data[index]
        input_ids, cu_seqlens, position_ids, labels, idxs = [], [0], [], [], []

        for pair in pairs:
            input_ids.extend(pair["chosen_ids"])
            input_ids.extend(pair["rejected_ids"])

            position_ids.extend(list(range(len(pair["chosen_ids"]))))
            position_ids.extend(list(range(len(pair["rejected_ids"]))))

            labels.extend(pair["chosen_labels"])
            labels.extend(pair["rejected_labels"])

            cu_seqlens.append(cu_seqlens[-1] + len(pair["chosen_ids"]))
            cu_seqlens.append(cu_seqlens[-1] + len(pair["rejected_ids"]))

            idxs.append(pair["idx"])

        return {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
            "cumulative_len": cu_seqlens,
            "raw_data_idx": idxs,
        }


class PackedDatasetWrapperStream(IterableDataset):
    def __init__(
        self,
        dataset,              # 这里应该是一个可迭代的数据源，或另一个 IterableDataset
        max_packed_length=16384,
        avg_num_per_pack=5,
    ):
        super().__init__()
        self.dataset = dataset
        self.max_packed_length = max_packed_length

        # hard coded!  # noqa
        self.avg_num_per_pack = avg_num_per_pack

        self.data_num = int(len(dataset) // self.avg_num_per_pack)

    def __len__(self):
        return self.data_num

    def __iter__(self):
        """
        边读边打包，将若干条数据拼接成一个“bin”，
        一旦长度超过上限，就 yield 当前 bin 并开启新 bin。
        """
        data_bin = []
        bin_seq_len = 0
        skipped = 0  # 用于统计有多少样本因为过长被直接跳过

        # 1. 从上游 dataset 持续读取数据
        for pair in self.dataset:
            # 计算当前 pair 的总长度
            cur_len = len(pair["chosen_ids"]) + len(pair["rejected_ids"])

            # 如果这条数据本身就大于 max_packed_length，就跳过（或记录日志后跳过）
            if cur_len > self.max_packed_length:
                skipped += 1
                continue

            # 如果当前 bin + 新样本会超限，则先把已有 bin 产出
            if (bin_seq_len + cur_len) > self.max_packed_length and len(data_bin) > 0:
                yield self._convert_bin_to_dict(data_bin)

                # 重置 bin
                data_bin = []
                bin_seq_len = 0

            # 将新数据放入 bin
            data_bin.append(pair)
            bin_seq_len += cur_len

        # 2. 所有数据读取完后，如果 bin 里仍有数据，最终再产出一次
        if len(data_bin) > 0:
            yield self._convert_bin_to_dict(data_bin)

        # 3. 如果需要，可以在最后打印或记录被跳过的样本数（可选）
        if skipped > 0:
            print_log(
                f"{skipped} samples were skipped because "
                f"their length was larger than {self.max_packed_length}.",
                logger="current",
            )
        print_log(
            f"The batch numbers of dataset is changed "
            f"from {len(self.dataset)} to {len(self)} after"
            " using var len attention.",
            logger="current",
        )

    def _convert_bin_to_dict(self, pairs):
        """
        将一个 bin（其中包含多条 pair）转换成原先 __getitem__ 中拼接后的格式。
        这里的逻辑与原始的 __getitem__ 保持一致即可：
          - 拼接 chosen_ids 和 rejected_ids 为 input_ids
          - 同样处理 position_ids 和 labels
          - 维护一个 cu_seqlens（cumulative_len）数组
        """
        input_ids = []
        position_ids = []
        labels = []
        cu_seqlens = [0]
        idxs = []

        for pair in pairs:
            chosen_len = len(pair["chosen_ids"])
            rejected_len = len(pair["rejected_ids"])

            input_ids.extend(pair["chosen_ids"])
            input_ids.extend(pair["rejected_ids"])

            position_ids.extend(range(chosen_len))
            position_ids.extend(range(rejected_len))

            labels.extend(pair["chosen_labels"])
            labels.extend(pair["rejected_labels"])

            # cu_seqlens[-1] 是当前已累积的长度
            cu_seqlens.append(cu_seqlens[-1] + chosen_len)
            cu_seqlens.append(cu_seqlens[-1] + rejected_len)

            idxs.append(pair["idx"])

        return {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
            "cumulative_len": cu_seqlens,
            "raw_data_idx": idxs,
        }


def unpack_seq(seq, cu_seqlens):
    """Unpack a packed sequence to a list of sequences with different
    lengths."""
    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    subseqs = seq.split(seqlens)
    return subseqs


def broad_cast_dataset(dataset):
    xtuner_dataset_timeout = timedelta(
        minutes=int(os.getenv("XTUNER_DATASET_TIMEOUT", default=60))
    )
    print_log(f"xtuner_dataset_timeout = {xtuner_dataset_timeout}", logger="current")
    using_dist = dist.is_available() and dist.is_initialized()
    if using_dist:
        # monitored barrier requires gloo process group to perform host-side sync.  # noqa
        group_gloo = dist.new_group(backend="gloo", timeout=xtuner_dataset_timeout)
    if not using_dist or dist.get_rank() == 0:
        objects = [dataset]
    else:
        objects = [None]
    if using_dist:
        dist.monitored_barrier(group=group_gloo, timeout=xtuner_dataset_timeout)
        dist.broadcast_object_list(objects, src=0)
    return objects[0]


def map_dataset(dataset, dataset_map_fn, map_num_proc):
    if isinstance(dataset_map_fn, str):
        map_fn_obj = MAP_FUNC.get(dataset_map_fn) or get_object_from_string(
            dataset_map_fn
        )
        if map_fn_obj is not None:
            dataset_map_fn = map_fn_obj
        else:
            raise TypeError(
                "dataset_map_fn must be a function or a "
                "registered function's string in MAP_FUNC, "
                f"but got a string of '{dataset_map_fn}'"
            )

    dataset = dataset.map(dataset_map_fn, num_proc=map_num_proc)
    return dataset


def build_preference_dataset(
    dataset: str,
    tokenizer: AutoTokenizer,
    max_length: int,
    max_response_length: int,
    dataset_map_fn: Callable = None,
    is_dpo: bool = True,
    is_reward: bool = False,
    reward_token_id: int = -1,
    num_proc: int = 32,
    use_varlen_attn: bool = False,
    max_packed_length: int = 16384,
    shuffle_before_pack: bool = True,
) -> Dataset:
    using_dist = dist.is_available() and dist.is_initialized()
    tokenized_ds = None
    if not using_dist or dist.get_rank() == 0:
        if (
            isinstance(tokenizer, dict)
            or isinstance(tokenizer, Config)
            or isinstance(tokenizer, ConfigDict)
        ):
            tokenizer = BUILDER.build(tokenizer)

        dataset = build_origin_dataset(dataset, split="train")
        if dataset_map_fn is not None:
            dataset = map_dataset(dataset, dataset_map_fn, map_num_proc=num_proc)

        tokenized_ds = PreferenceDataset(
            dataset=dataset,
            tokenizer=tokenizer,
            max_length=max_length,
            max_response_length=max_response_length,
            is_dpo=is_dpo,
            is_reward=is_reward,
            reward_token_id=reward_token_id,
            num_proc=num_proc,
        )
        if use_varlen_attn:
            tokenized_ds = PackedDatasetWrapper(
                dataset=tokenized_ds,
                max_packed_length=max_packed_length,
                shuffle_before_pack=shuffle_before_pack,
            )
    tokenized_ds = broad_cast_dataset(tokenized_ds)
    return tokenized_ds


def build_preference_dataset_stream(
    dataset: str,
    tokenizer: AutoTokenizer,
    max_length: int,
    max_response_length: int,
    dataset_map_fn: Callable = None,
    is_dpo: bool = True,
    is_reward: bool = False,
    reward_token_id: int = -1,
    num_proc: int = 32,
    use_varlen_attn: bool = False,
    max_packed_length: int = 16384,
    avg_num_per_pack: int = 5,
    shuffle_before_pack: bool = True,
    data_num: int = 0,
    if_pretrain: bool = False,
    work_dir_name: str = None,
) -> Dataset:

    using_dist = dist.is_available() and dist.is_initialized()
    if using_dist:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if (
        isinstance(tokenizer, dict)
        or isinstance(tokenizer, Config)
        or isinstance(tokenizer, ConfigDict)
    ):
        tokenizer = BUILDER.build(tokenizer)

    if if_pretrain:
        dataset = PretrainPreferenceDatasetStream(
            dataset["path"],
            rank=rank,
            world_size=world_size,
            work_dir_name=work_dir_name,
        )
    else:
        dataset = build_origin_dataset(dataset, split="train")

    if dataset_map_fn is not None:
        dataset = map_dataset(dataset, dataset_map_fn, map_num_proc=num_proc)

    tokenized_ds = PreferenceDatasetStream(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=max_length,
        max_response_length=max_response_length,
        is_dpo=is_dpo,
        is_reward=is_reward,
        reward_token_id=reward_token_id,
        data_num=data_num,
        rank=rank,
        world_size=world_size,
        use_varlen_attn=use_varlen_attn,
    )
    if use_varlen_attn:
        tokenized_ds = PackedDatasetWrapperStream(
            dataset=tokenized_ds,
            max_packed_length=max_packed_length,
            avg_num_per_pack=avg_num_per_pack,
        )
    return tokenized_ds


def intel_orca_dpo_map_fn(example):
    prompt = [
        {"role": "system", "content": example["system"]},
        {"role": "user", "content": example["question"]},
    ]
    chosen = [{"role": "assistant", "content": example["chosen"]}]
    rejected = [{"role": "assistant", "content": example["rejected"]}]
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def orpo_dpo_mix_40k_map_fn(example):
    assert len(example["chosen"]) == len(example["rejected"])
    prompt = example["chosen"][:-1]
    chosen = example["chosen"][-1:]
    rejected = example["rejected"][-1:]
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}
