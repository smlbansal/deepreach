import torch
from utils import diff_operators, quaternion

# uses real units

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def init_brt_hjivi_loss(
    dynamics,
    minWith,
    dirichlet_loss_divisor,
    MPC_loss_type,
    use_MPC,
    MPC_finetune_lambda,
):
    def brt_hjivi_loss(
        state,
        value,
        dvdt,
        dvds,
        boundary_value,
        dirichlet_mask,
        output,
        MPC_values,
        MPC_labels,
        use_MPC_terminal_loss=False,
    ):
        # Curriculum training loss
        if dynamics.deepReach_model == "exact":
            dirichlet_loss = torch.Tensor([0]).to(device)
        else:
            dirichlet = value[dirichlet_mask] - boundary_value[dirichlet_mask]
            dirichlet_loss = torch.abs(dirichlet).sum() / dirichlet_loss_divisor
        if MPC_loss_type == "l1":
            mpc_loss = torch.abs(MPC_values - MPC_labels).sum()
            if use_MPC_terminal_loss:
                # scale critical values
                if dynamics.set_mode == "avoid":
                    unsafe_cost_safe_value_indeces = (
                        torch.argwhere(
                            torch.logical_and(
                                MPC_labels.squeeze(0) < 0.0,
                                MPC_values.squeeze(0) >= 0.0,
                            )
                        )
                        .detach()
                        .squeeze(-1)
                    )
                else:
                    unsafe_cost_safe_value_indeces = (
                        torch.argwhere(
                            torch.logical_and(
                                MPC_labels.squeeze(0) > 0.0,
                                MPC_values.squeeze(0) <= 0.0,
                            )
                        )
                        .detach()
                        .squeeze(-1)
                    )
                mpc_loss += (
                    torch.abs(
                        MPC_values[:, unsafe_cost_safe_value_indeces]
                        - MPC_labels[:, unsafe_cost_safe_value_indeces]
                    )
                    * torch.abs(MPC_values[:, unsafe_cost_safe_value_indeces])
                ).sum() * MPC_finetune_lambda

        elif MPC_loss_type == "l2":
            mpc_loss = torch.pow(MPC_values - MPC_labels, 2).sum()
            # scale critical values
            if use_MPC_terminal_loss:
                if dynamics.set_mode == "avoid":
                    unsafe_cost_safe_value_indeces = (
                        torch.argwhere(
                            torch.logical_and(
                                MPC_labels.squeeze(0) < 0.0,
                                MPC_values.squeeze(0) >= 0.0,
                            )
                        )
                        .detach()
                        .squeeze(-1)
                    )
                else:
                    unsafe_cost_safe_value_indeces = (
                        torch.argwhere(
                            torch.logical_and(
                                MPC_labels.squeeze(0) > 0.0,
                                MPC_values.squeeze(0) <= 0.0,
                            )
                        )
                        .detach()
                        .squeeze(-1)
                    )

                mpc_loss += (
                    torch.pow(
                        MPC_values[:, unsafe_cost_safe_value_indeces]
                        - MPC_labels[:, unsafe_cost_safe_value_indeces],
                        2,
                    )
                    * torch.abs(MPC_values[:, unsafe_cost_safe_value_indeces])
                ).sum() * MPC_finetune_lambda
        else:
            raise NotImplementedError

        if torch.all(dirichlet_mask):  # pretraining loss
            diff_constraint_hom = torch.Tensor([0]).to(device)
            if use_MPC and MPC_values.ndim > 1:
                dirichlet_loss += mpc_loss * 0.3
            elif dynamics.deepReach_model == "exact":
                dirichlet = output.squeeze(dim=-1)[dirichlet_mask] - 0.0
                dirichlet_loss = torch.abs(dirichlet).sum()
        else:
            ham = dynamics.hamiltonian(state, dvds)
            if minWith == "zero":
                ham = torch.clamp(ham, max=0.0)

            diff_constraint_hom = dvdt - ham
            if minWith == "target":
                diff_constraint_hom = torch.max(
                    diff_constraint_hom, value - boundary_value
                )
            # diff_constraint_hom = dvdt + ham # for FRS

        return {
            "dirichlet": dirichlet_loss,
            "diff_constraint_hom": torch.abs(diff_constraint_hom).sum(),
            "mpc_loss": mpc_loss,
        }

    return brt_hjivi_loss


def init_brat_hjivi_loss(
    dynamics,
    minWith,
    dirichlet_loss_divisor,
    MPC_loss_type,
    use_MPC,
    MPC_finetune_lambda,
):
    def brat_hjivi_loss(
        state,
        value,
        dvdt,
        dvds,
        boundary_value,
        reach_value,
        avoid_value,
        dirichlet_mask,
        output,
        MPC_values,
        MPC_labels,
        use_MPC_terminal_loss=False,
    ):
        # Curriculum training loss
        if dynamics.deepReach_model == "exact":
            dirichlet_loss = torch.Tensor([0]).to(device)
        else:
            dirichlet = value[dirichlet_mask] - boundary_value[dirichlet_mask]
            dirichlet_loss = torch.abs(dirichlet).sum() / dirichlet_loss_divisor
        if MPC_loss_type == "l1":
            mpc_loss = torch.abs(MPC_values - MPC_labels).sum()
            if use_MPC_terminal_loss:
                # scale critical values
                unsafe_cost_safe_value_indeces = (
                    torch.argwhere(
                        torch.logical_and(
                            MPC_labels.squeeze(0) > 0.0, MPC_values.squeeze(0) <= 0.0
                        )
                    )
                    .detach()
                    .squeeze(-1)
                )
                mpc_loss += (
                    torch.abs(
                        MPC_values[:, unsafe_cost_safe_value_indeces]
                        - MPC_labels[:, unsafe_cost_safe_value_indeces]
                    )
                    * torch.abs(MPC_values[:, unsafe_cost_safe_value_indeces])
                ).sum() * MPC_finetune_lambda
        elif MPC_loss_type == "l2":
            mpc_loss = torch.pow(MPC_values - MPC_labels, 2).sum()
            if use_MPC_terminal_loss:
                unsafe_cost_safe_value_indeces = (
                    torch.argwhere(
                        torch.logical_and(
                            MPC_labels.squeeze(0) > 0.0, MPC_values.squeeze(0) <= 0.0
                        )
                    )
                    .detach()
                    .squeeze(-1)
                )

                mpc_loss += (
                    torch.pow(
                        MPC_values[:, unsafe_cost_safe_value_indeces]
                        - MPC_labels[:, unsafe_cost_safe_value_indeces],
                        2,
                    )
                    * torch.abs(MPC_values[:, unsafe_cost_safe_value_indeces])
                ).sum() * MPC_finetune_lambda
        else:
            raise NotImplementedError

        if torch.all(dirichlet_mask):  # pretraining loss
            diff_constraint_hom = torch.Tensor([0]).to(device)
            if use_MPC:
                dirichlet_loss += mpc_loss * 0.3
            elif dynamics.deepReach_model == "exact":
                dirichlet = output.squeeze(dim=-1)[dirichlet_mask] - 0.0
                dirichlet_loss = torch.abs(dirichlet).sum()
        else:
            ham = dynamics.hamiltonian(state, dvds)
            if minWith == "zero":
                ham = torch.clamp(ham, max=0.0)

            diff_constraint_hom = dvdt - ham
            if minWith == "target":
                diff_constraint_hom = torch.min(
                    torch.max(diff_constraint_hom, value - reach_value),
                    value + avoid_value,
                )

        return {
            "dirichlet": dirichlet_loss,
            "diff_constraint_hom": torch.abs(diff_constraint_hom).sum(),
            "mpc_loss": mpc_loss,
        }

    return brat_hjivi_loss
