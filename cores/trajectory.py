import torch
import torch.nn as nn

class Trajectory(nn.Module):
    """Class for recording and storing trajectory data."""

    def __init__(self):
        super().__init__()
        self.tensor_data = {}
        self.value_data = {}
        self._compile = False

    def add_tensor(self, name, images):
        """
            Adds image data to the trajectory.

            Parameters:
                name (str): Name of the image data.
                images (torch.Tensor): Image tensor to add.
        """
        if name not in self.tensor_data:
            self.tensor_data[name] = []
        self.tensor_data[name].append(images.detach().cpu())

    def add_value(self, name, values):
        """
            Adds value data to the trajectory.

            Parameters:
                name (str): Name of the value data.
                values (any): Value to add.
        """
        if name not in self.value_data:
            self.value_data[name] = []
        self.value_data[name].append(values)

    def compile(self):
        """
            Compiles the recorded data into tensors.

            Returns:
                Trajectory: The compiled trajectory object.
        """
        if not self._compile:
            self._compile = True
            for name in self.tensor_data.keys():
                self.tensor_data[name] = torch.stack(self.tensor_data[name], dim=0)
            for name in self.value_data.keys():
                self.value_data[name] = torch.tensor(self.value_data[name])
        return self

    @classmethod
    def merge(cls, trajs):
        """
        Merge a list of compiled trajectories from different batches.
        Supports:
            1) List[Trajectory]
            2) List[Dict[str, Trajectory]]
        """

        # -------------------------------
        # 情况 1：老版本，单 trajectory
        # -------------------------------
        if isinstance(trajs[0], cls):
            merged_traj = cls()
            for name in trajs[0].tensor_data.keys():
                merged_traj.tensor_data[name] = torch.cat(
                    [traj.tensor_data[name] for traj in trajs], dim=1
                )
            for name in trajs[0].value_data.keys():
                merged_traj.value_data[name] = trajs[0].value_data[name]
            return merged_traj

        # --------------------------------
        # 情况 2：新版本，多模型 trajectory
        # trajs: List[ Dict[str, Trajectory] ]
        # --------------------------------
        elif isinstance(trajs[0], dict):
            merged = {}
            keys = trajs[0].keys()

            for k in keys:
                model_trajs = [traj_dict[k] for traj_dict in trajs]

                merged_traj = cls()
                for name in model_trajs[0].tensor_data.keys():
                    merged_traj.tensor_data[name] = torch.cat(
                        [traj.tensor_data[name] for traj in model_trajs], dim=1
                    )
                for name in model_trajs[0].value_data.keys():
                    merged_traj.value_data[name] = model_trajs[0].value_data[name]

                merged[k] = merged_traj

            return merged

        else:
            raise TypeError("Unsupported trajectory format in merge()")
