import os


class GPUInfo():
    def __init__(self):
        self.gpu_info = []

    def get_gpu_info(self, sort_flag=False):
        """
        根据显卡 id 获取显存使用信息, 单位 GB
        :param gpu_id: 显卡 ID, 默认遍历所有显卡
        :return: total 所有的显存, used 当前使用的显存, free 可使用的显存
        """
        inf = os.popen('nvidia-smi | grep %').read().split('\n')[0:-1]
        self.gpu_info = []
        for index, line in enumerate(inf, start=0):
            line = line.split(' ')
            new_line = [index]
            for str1 in line:
                if 'C' in str1:
                    str1 = str1.replace('C', '')
                    new_line.append(int(str1))
                if 'MiB' in str1:
                    str1 = str1.replace('MiB', '')
                    new_line.append(int(str1) / 1024.0)
            self.gpu_info.append(new_line)

        if sort_flag:
            self._sort_memory_info()
        return self.gpu_info

    def get_topk(self, topk=0):
        inf = self.get_gpu_info(sort_flag=True)
        gpu_rank = []
        if isinstance(topk, int) and topk > 0:
            if topk > len(inf):
                raise Exception('topk超过显卡实际数量')
            else:
                for i in range(topk):
                    gpu_rank.append(inf[i][0])

        return gpu_rank

    def _sort_memory_info(self):
        self.gpu_info.sort(key=lambda x: x[3] - x[2], reverse=True)

    def __call__(self, sort=False, show=False):
        self.gpu_info = self.get_gpu_info(sort)
        left = self.gpu_info[0][3] - self.gpu_info[0][2]
        if show:
            print('显卡{} 内存剩余{:.2f}G'.format(self.gpu_info[0][0], left))
        return self.gpu_info


if __name__ == '__main__':
    gpu_info = GPUInfo().get_topk(topk=7)
    print(str(gpu_info))
