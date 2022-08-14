from yacs.config import CfgNode
import yaml
import os
import git


def load_config(path):
    with open(path, "r") as f:
        config = CfgNode(yaml.load(f, Loader=yaml.Loader))
    return config


def save_yaml(dir, cfg):
    with open(dir + '/config.yaml', 'w') as f:
        f.write(yaml.dump(cfg))


def get_next_exp_number(root_dir):
    rootdir = 'experiments'
    exp_num = []
    if not os.path.isdir(root_dir):
        print(f"No experiment folder found, root directory: {rootdir}")
        exit(0)
    for file in os.listdir(rootdir):
        d = os.path.join(rootdir, file)
        if os.path.isdir(d):
            exp_num.append(int(''.join([i for i in file if i.isdigit()])))
    if not exp_num:
        return 1
    return max(exp_num)+1


def make_exp_Folder(env_name=None, tb_dir_flag=False):
    exp_num = get_next_exp_number('./experiments/')
    if env_name is None:
        exp_dir = './experiments/exp' + str(exp_num)
    else:
        exp_dir = './experiments/' + str(env_name)
    os.mkdir(exp_dir)
    model_dir = exp_dir + '/model'
    os.mkdir(model_dir)
    tb_dir = 'exp' + str(exp_num)
    if tb_dir_flag:
        return [exp_dir, model_dir,  tb_dir]
    return exp_dir, model_dir


def get_git_hash(cfg):
    repo = git.Repo(search_parent_directories=True)
    diff = None
    if repo.is_dirty():
        print(f"Repo in {repo.working_dir} has uncommited changes. Trying to create a diff file ...")
        try:
            diff = repo.git.diff(repo.head.commit.tree) + '\n'
        except Exception as e:
            print(f"Couldn't create diff file: {e}")
            exit(0)
    cfg.PROJECT_GITHASH = None  # repo.head.object.hexsha
    cfg.PROJECT_ROOT = repo.git_dir[:-4]

    try:
        cfg.PROJECT_REMOTE = list(repo.remote("origin").urls)[0]
        # assumes origin as default remote and only one linked url
    except Exception as e:
        print(f"Error during remote retrieval. If you rerun, pektorch will use the project's root instead. Error: {e}")

    # cfg.PEKTORCH_GITHASH = get_pektorch_hash()
    cfg.PROJECT_GIT_DIFF = diff
    return cfg, diff


if __name__ == "__main__":
    print(load_config("./config/base.yaml"))
    exp, _ = make_exp_Folder()
    print(exp)
    print(get_next_exp_number("./experiments"))
