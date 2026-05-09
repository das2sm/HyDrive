import pickle
import os
import numpy as np

def compare_pkl_files(file1_path, file2_path, token_key='token', verbose=True):
    """
    比较两个pkl文件的数据长度和每一帧的token是否一致
    
    参数:
    file1_path (str): 第一个pkl文件路径
    file2_path (str): 第二个pkl文件路径
    token_key (str): 包含token的字典键名
    verbose (bool): 是否打印详细比较结果
    
    返回:
    bool: 所有数据是否完全一致
    """
    # 检查文件是否存在
    for path in [file1_path, file2_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件不存在: {path}")
    
    # 加载pkl文件数据
    with open(file1_path, 'rb') as f:
        data1 = pickle.load(f)
    with open(file2_path, 'rb') as f:
        data2 = pickle.load(f)
    
    # data1 = list(sorted(data1, key=lambda e: e["timestamp"]))
    # data2 = list(sorted(data2, key=lambda e: e["timestamp"]))
    data1 = list(sorted(data1, key=lambda e: e["token"]))
    data2 = list(sorted(data2, key=lambda e: e["token"]))
    
    # 检查数据类型是否一致
    if type(data1) != type(data2):
        print(f"数据类型不一致: {type(data1)} vs {type(data2)}")
        return False
    
    # 1. 比较数据长度
    len_match = len(data1) == len(data2)
    if verbose:
        print(f"文件1帧数: {len(data1)}, 文件2帧数: {len(data2)}, 长度一致: {'是' if len_match else '否'}")
    
    if not len_match:
        return False
    
    # 2. 逐帧比较token
    all_tokens_match = True
    mismatch_count = 0
    frame_diffs = []
    
    for i in range(len(data1)):
        print(i)
        frame1 = data1[i]
        frame2 = data2[i]
        
        # 检查帧数据类型
        if type(frame1) != type(frame2):
            print(f"第 {i} 帧数据类型不一致: {type(frame1)} vs {type(frame2)}")
            all_tokens_match = False
            mismatch_count += 1
            continue
        
        # 如果是字典类型且包含token键
        if isinstance(frame1, dict) and token_key in frame1 and token_key in frame2:
            token1 = frame1[token_key]
            token2 = frame2[token_key]
            
            # 判断token是否一致
            token_match = token1 == token2
            
            if not token_match:
                if verbose:
                    print(f"帧 {i} Token不一致:\n文件1: {token1}\n文件2: {token2}")
                all_tokens_match = False
                mismatch_count += 1
                frame_diffs.append(i)
        
        else:
            print(f"第 {i} 帧不包含有效的token数据")
            all_tokens_match = False
            mismatch_count += 1
        
        def compare(a, b):
            # try:
            if type(a) is dict:
                for key in a.keys():
                    if not compare(a[key], b[key]):
                        return False
                return True
            
            if type(a) is list:
                for j in range(len(a)):
                    if not compare(a[j], b[j]):
                        return False
                return True

            if type(a) is np.ndarray:
                if (a==b).all() or np.allclose(a, b):
                    return True
            if a == b:
                return True
            return False
            # except:
            #     import ipdb; ipdb.set_trace()
            #     print()
        
        assert compare(frame1, frame2)

        # for key in frame1.keys():
        #     print(key)
        #     # if key == "command_far_xy":
        #     #     import ipdb; ipdb.set_trace()
        #     if type(frame1[key]) is np.ndarray:
        #         try:
        #             assert np.allclose(frame1[key],frame2[key])
        #         except:
        #             import ipdb; ipdb.set_trace()
        #     elif type(frame1[key]) is dict:
        #         continue
        #     else:
        #         assert frame1[key] == frame2[key], print(key)

    # 打印总结报告
    if verbose:
        print("\n===== 比较结果总结 =====")
        print(f"总帧数: {len(data1)}")
        print(f"Token一致帧数: {len(data1) - mismatch_count}/{len(data1)}")
        print(f"Token不一致帧数: {mismatch_count}")
        
        if frame_diffs:
            print(f"不一致的帧索引 (最多显示10个): {frame_diffs[:10]}{'...' if len(frame_diffs) > 10 else ''}")
        
        print(f"所有Token是否一致: {'是' if all_tokens_match else '否'}")
        print(f"文件是否完全一致: {'是' if len_match and all_tokens_match else '否'}")
    
    return len_match and all_tokens_match

if __name__ == "__main__":
    # 使用示例
    file1 = "/horizon-bucket/HCT_Perception2/users/wenchao01.sun/SparseDrive/data/infos/b2d_infos_train.pkl"
    file2 = "data/infos_b2d/b2d_infos_train.pkl"
    
    # file1 = "data/infos/b2d_infos_val.pkl"
    # file2 = "data/infos_b2d/b2d_infos_val.pkl"

    result = compare_pkl_files(file1, file2)
    print(f"\n最终结果: {'文件完全一致' if result else '文件存在差异'}")
    # except Exception as e:
    #     print(f"比较过程中出错: {str(e)}")