from mmengine.hooks import Hook
from mmengine.fileio import FileClient, get_file_backend
from mmengine import mkdir_or_exist
from typing import Optional, Union
from pathlib import Path
import os.path as osp
import torch.distributed as dist

DATA_BATCH = Optional[Union[dict, tuple, list]]


class DataResumeHook(Hook):
    """Hook to resume data from checkpoint.

    Args:
        data (DATA_BATCH): Data to be resumed.
    """
    priority = 'VERY_LOW'

    def __init__(self,
                 interval: int = -1,
                 by_epoch: bool = True,
                 save_last: bool = True,
                 save_begin: int = 0,
                 backend_args: Optional[dict] = None,
                 file_client_args: Optional[dict] = None,
                 out_dir: Optional[Union[str, Path]] = None,
                 ) -> None:
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_last = save_last
        self.save_begin = save_begin
        self.backend_args = backend_args
        self.file_client_args = file_client_args
        self.out_dir = out_dir

    def before_train(self, runner) -> None:
        """Finish all operations, related to checkpoint.

        This function will get the appropriate file client, and the directory
        to save these checkpoints of the model.

        Args:
            runner (Runner): The runner of the training process.
        """
        if self.out_dir is None:
            self.out_dir = runner.work_dir

        # If self.file_client_args is None, self.file_client will not
        # used in CheckpointHook. To avoid breaking backward compatibility,
        # it will not be removed util the release of MMEngine1.0
        self.file_client = FileClient.infer_client(self.file_client_args,
                                                   self.out_dir)

        if self.file_client_args is None:
            self.file_backend = get_file_backend(
                self.out_dir, backend_args=self.backend_args)
        else:
            self.file_backend = self.file_client

        using_dist = dist.is_available() and dist.is_initialized()
        if using_dist:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        last_data = self.file_backend.join_path(self.out_dir,
                                                "last_ckpt_data_idxes")
        if self.rank == 0:
            mkdir_or_exist(last_data)

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs=Optional[dict]) -> None:
        """Save the checkpoint and synchronize buffers after each iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            outputs (dict, optional): Outputs from model.
        """
        if self.by_epoch:
            return

        # save checkpoint for following cases:
        # 1. every ``self.interval`` iterations
        #       which start at ``self.save_begin``
        # 2. reach the last iteration of training
        if self.every_n_train_iters(runner, self.interval,
                                    self.save_begin) or \
                (self.save_last and self.is_last_train_iter(runner)):
            runner.logger.info(
                f'Saving data indexes at {runner.iter + 1} iterations')
            self._save_data_indexes(data_batch['data'])

    def _save_data_indexes(self, data: dict) -> None:
        # remove other checkpoints before save checkpoint to make the
        # self.keep_ckpt_ids are saved as expected
        last_data = self.file_backend.join_path(self.out_dir,
                                                "last_ckpt_data_idxes")

        save_file = osp.join(last_data, str(self.rank))
        with open(save_file, 'w') as f:
            f.write(str(data["data_idxes"]))  # type: ignore
