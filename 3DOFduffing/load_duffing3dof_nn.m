function [net, inputMean, inputStd, targetMean, targetStd] = load_duffing3dof_nn()
%LOAD_DUFFING3DOF_NN  加载从 Mathematica 导出的 3DOF Duffing MLP 和归一化参数
%
% 输出:
%   net        - 导入的回归网络
%   inputMean  - 1x7 输入均值向量
%   inputStd   - 1x7 输入标准差向量
%   targetMean - 1x3 输出均值向量
%   targetStd  - 1x3 输出标准差向量

    %% 1) 导入 ONNX 网络
    net = importONNXNetwork( ...
              "trainedNet_3dof_parametric_final.onnx", ...
              "OutputLayerType","regression", ...
              "InputDataFormats","BC");  % 输入: batch x feature

    %% 2) 导入归一化参数（.mat）
    S = load("normParams_3dof_parametric_final.mat");

    fns = fieldnames(S);

    if ismember("inputMean", fns)
        % ---- 情况 A：Mathematica 直接导出了 4 个变量 ----
        inputMeanStruct  = S;
    else
        % ---- 情况 B：只有一个总变量，里面再存 Association / struct ----
        if numel(fns) ~= 1
            error("无法在 normParams_3dof_parametric_final.mat 中找到 inputMean 等字段。顶层字段: %s", strjoin(fns.', ', '));
        end
        top = S.(fns{1});   % 比如 S.normParams_3dof_parametric_final

        % 兼容 struct / containers.Map / struct-like
        if isstruct(top)
            inputMeanStruct = top;
        else
            error("顶层变量 %s 不是 struct，无法提取 inputMean 等字段。", fns{1});
        end
    end

    % 现在从 inputMeanStruct 里取字段
    if ~isfield(inputMeanStruct, "inputMean") || ...
       ~isfield(inputMeanStruct, "inputStd")  || ...
       ~isfield(inputMeanStruct, "targetMean")|| ...
       ~isfield(inputMeanStruct, "targetStd")
        error("在归一化参数结构中找不到 inputMean/inputStd/targetMean/targetStd 字段。");
    end

    inputMean  = inputMeanStruct.inputMean(:).';   % 转成 1x7
    inputStd   = inputMeanStruct.inputStd(:).';
    targetMean = inputMeanStruct.targetMean(:).';  % 转成 1x3
    targetStd  = inputMeanStruct.targetStd(:).';
end
