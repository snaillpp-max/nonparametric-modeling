%% simulate_duffing3_mlp_from_wl.m 
% 3DOF 参数激励 Duffing：MLP（来自 Mathematica） vs 真 ODE
% 状态: y = [x1,x2,x3,v1,v2,v3]^T
% MLP 输入: [x1,x2,x3,v1,v2,v3,kappa_eff]
% MLP 输出: [ga1,ga2,ga3]
% 外部强迫项仅在 x1'' 上加：+Fext*cos(omegaF*t)（不进 NN 输入）
% 庞加莱截面规则：
%   若存在外激励 (Fext ~= 0 && omegaF ~= 0)，则按外激励周期取截面；
%   否则按参数激励周期取截面。
% 庞加莱截面对每个自由度分别在 (xi, vi) 平面上绘制。
close all;  % 关闭所有图形窗口
clear all;  % 清除所有变量（包括持久变量）
clc;        % 清空命令窗口
clear; clc; rng('shuffle');

fprintf('========== 3DOF Parametric Duffing + MLP（Mathematica） ==========\n');

%% ---------------------------------------------------------
% 1. 加载 MLP（ONNX + 归一化参数）
%% ---------------------------------------------------------
[net_pd3, inMean_pd3, inStd_pd3, tarMean_pd3, tarStd_pd3] = load_duffing3dof_nn();

net     = net_pd3;
inMean  = inMean_pd3;
inStd   = inStd_pd3;
outMean = tarMean_pd3;
outStd  = tarStd_pd3;

% 防止除以 0
epsStd  = 1e-8;
inStd   = max(inStd, epsStd);
outStd  = max(outStd, epsStd);

normalizeInput    = @(x) (x - inMean) ./ inStd;
denormalizeOutput = @(yn) yn .* outStd + outMean;

fprintf('已加载 MLP（来自 Mathematica ONNX）。\n');
fprintf('输入维度 = %d, 输出维度 = %d\n', numel(inMean), numel(outMean));

%% ---------------------------------------------------------
% 2. 物理参数（与 Mathematica 训练一致）
%% ---------------------------------------------------------
alpha = -1;
beta  =  1;
c     =  0.3;
kc    =  1;

fprintf('物理参数: alpha=%g, beta=%g, c=%g, kc=%g\n', alpha, beta, c, kc);

%% ---------------------------------------------------------
% 3. 参数激励 + 强迫激励参数
%% ---------------------------------------------------------
kappa  = 0.5;     % 参数激励幅值
omegaP = 1.0;     % 参数激励频率 (rad/s)

Fext   =0.6;     % 外部强迫振动幅值（0 表示无外激励）
omegaF =0.8;     % 外部强迫频率 (rad/s)，仅在 Fext ~= 0 时起作用

T_p  = 2*pi/omegaP;
tEnd = 50*T_p;

fprintf('参数激励: kappa=%g, Omega_p=%g rad/s\n', kappa, omegaP);
fprintf('外部强迫: Fext=%g, Omega_F=%g rad/s\n', Fext, omegaF);
fprintf('仿真时长: %.2f 秒（约 %.1f 个周期）\n', tEnd, tEnd/T_p);

%% ---------------------------------------------------------
% 4. 初值
%% ---------------------------------------------------------
x1_0 = 1;  x2_0 = 1;  x3_0 = 1;
v1_0 = 0;  v2_0 = 0;  v3_0 = 0;

y0 = [x1_0 x2_0 x3_0 v1_0 v2_0 v3_0]';
fprintf('初值: x=[%.2f %.2f %.2f], v=[%.2f %.2f %.2f]\n', y0);

%% ---------------------------------------------------------
% 5. 参数结构体
%% ---------------------------------------------------------
paramMLP.net               = net;
paramMLP.normalizeInput    = normalizeInput;
paramMLP.denormalizeOutput = denormalizeOutput;
paramMLP.kappa             = kappa;
paramMLP.omegaP            = omegaP;
paramMLP.Fext              = Fext;
paramMLP.omegaF            = omegaF;

paramTrue.alpha  = alpha;
paramTrue.beta   = beta;
paramTrue.c      = c;
paramTrue.kc     = kc;
paramTrue.kappa  = kappa;
paramTrue.omegaP = omegaP;
paramTrue.Fext   = Fext;
paramTrue.omegaF = omegaF;

%% ---------------------------------------------------------
% 6. 积分
%% ---------------------------------------------------------
tSpan = linspace(0, tEnd, 4001);
opts  = odeset("RelTol",1e-6, "AbsTol",1e-8);

fprintf('\n开始积分（MLP 模型）...\n');
[tMLP, yMLP] = ode45(@(t,y) duffing3_mlp_ode(t,y,paramMLP), tSpan, y0, opts);
fprintf('MLP 积分完成\n');

fprintf('\n开始积分（真实 ODE）...\n');
[tTrue, yTrue] = ode45(@(t,y) duffing3_true_ode(t,y,paramTrue), tSpan, y0, opts);
fprintf('真实 ODE 积分完成\n');

if any(abs(tMLP - tTrue) > 1e-10)
    yTrue = interp1(tTrue, yTrue, tMLP);
    t = tMLP;
else
    t = tMLP;
end

xMLP  = yMLP(:,1:3); vMLP  = yMLP(:,4:6);
xTrue = yTrue(:,1:3); vTrue = yTrue(:,4:6);

% 误差时间历程（位移）
errX = xMLP - xTrue;   % 每列分别对应 x1,x2,x3 的误差

%% ---------------------------------------------------------
% 7. 庞加莱截面（3 个自由度，各画一张）
%% ---------------------------------------------------------
% 规则：
%   若存在外激励 (Fext ~= 0 && omegaF ~= 0) → 用外激励周期；
%   否则用参数激励周期。
if Fext ~= 0 && omegaF ~= 0
    poincOmega = omegaF;
    fprintf('\n庞加莱截面：使用外激励频率 omegaF=%.4f rad/s\n', omegaF);
else
    poincOmega = omegaP;
    fprintf('\n庞加莱截面：使用参数激励频率 omegaP=%.4f rad/s\n', omegaP);
end

poincT = 2*pi / poincOmega;     % 采样周期（秒）
poincTransient = 100;           % 前 100 s 视为瞬态，丢弃（可按需修改）

if poincTransient >= tEnd
    poincTransient = 0.3 * tEnd;   % 防呆：如果太长，改成 0.3 tEnd
end

tPoinc = (poincTransient : poincT : tEnd).';   % 采样时刻列向量

if isempty(tPoinc)
    warning('庞加莱截面: tPoinc 为空，模拟时间可能太短。');
    xTrueP = [];
    vTrueP = [];
    xMLPP  = [];
    vMLPP  = [];
else
    % 对 3 个自由度分别在 (xi, vi) 平面上取截面
    nP = numel(tPoinc);
    xTrueP = zeros(nP,3);
    vTrueP = zeros(nP,3);
    xMLPP  = zeros(nP,3);
    vMLPP  = zeros(nP,3);

    for i = 1:3
        xTrueP(:,i) = interp1(t, xTrue(:,i), tPoinc);
        vTrueP(:,i) = interp1(t, vTrue(:,i), tPoinc);
        xMLPP(:,i)  = interp1(t, xMLP(:,i),  tPoinc);
        vMLPP(:,i)  = interp1(t, vMLP(:,i),  tPoinc);
    end

    fprintf('庞加莱截面：T=%.4f s，丢弃前 %.2f s，截面点数 = %d\n', ...
        poincT, poincTransient, numel(tPoinc));
end

%% ---------------------------------------------------------
% 8. 绘图：时间历程 + 误差
%% ---------------------------------------------------------
figure('Name','3DOF Duffing: MLP vs True','Position',[100 100 1100 800]);

subplot(3,2,1); hold on; grid on;
plot(t,xTrue(:,1),'k','LineWidth',1.2);
plot(t,xMLP(:,1),'r--','LineWidth',1.0);
title('x_1(t)'); legend('True','MLP');

subplot(3,2,2); grid on;
plot(t,errX(:,1),'b'); title('x_1(t) 误差');

subplot(3,2,3); hold on; grid on;
plot(t,xTrue(:,2),'k','LineWidth',1.2);
plot(t,xMLP(:,2),'r--','LineWidth',1.0);
title('x_2(t)'); legend('True','MLP');

subplot(3,2,4); grid on;
plot(t,errX(:,2),'b'); title('x_2(t) 误差');

subplot(3,2,5); hold on; grid on;
plot(t,xTrue(:,3),'k','LineWidth',1.2);
plot(t,xMLP(:,3),'r--','LineWidth',1.0);
title('x_3(t)'); legend('True','MLP');

subplot(3,2,6); grid on;
plot(t,errX(:,3),'b'); title('x_3(t) 误差');

%% ---------------------------------------------------------
% 9. 相图
%% ---------------------------------------------------------
figure('Name','相图对比','Position',[200 100 900 700]);

subplot(2,2,1); hold on; grid on;
plot(xTrue(:,1),vTrue(:,1),'k');
plot(xMLP(:,1),vMLP(:,1),'r--');
title('(x_1,v_1) 相图'); legend('True','MLP');

subplot(2,2,2); hold on; grid on;
plot(xTrue(:,2),vTrue(:,2),'k');
plot(xMLP(:,2),vMLP(:,2),'r--');
title('(x_2,v_2) 相图');

subplot(2,2,3); hold on; grid on;
plot(xTrue(:,3),vTrue(:,3),'k');
plot(xMLP(:,3),vMLP(:,3),'r--');
title('(x_3,v_3) 相图');

subplot(2,2,4); hold on; grid on;
plot3(xTrue(:,1),xTrue(:,2),xTrue(:,3),'k');
plot3(xMLP(:,1),xMLP(:,2),xMLP(:,3),'r--');
title('三维轨迹对比'); legend('True','MLP');

%% ---------------------------------------------------------
% 10. 庞加莱截面图（3 个自由度）
%% ---------------------------------------------------------
if ~isempty(tPoinc)
    figure('Name','Poincare Sections (3 DOFs)','Position',[300 100 1200 400]);

    dofNames = {'1','2','3'};
    for i = 1:3
        subplot(1,3,i); hold on; grid on;
        plot(xTrueP(:,i), vTrueP(:,i), 'ko', 'MarkerSize', 5, 'MarkerFaceColor', 'k');
        plot(xMLPP(:,i),  vMLPP(:,i),  'ro', 'MarkerSize', 4);
        xlabel(sprintf('x_%d', i));
        ylabel(sprintf('v_%d', i));
        title(sprintf('DOF %d Poincare (T = %.4f s)', i, poincT));
        if i == 1
            legend('True','MLP','Location','best');
        end
    end
else
    fprintf('庞加莱截面：无有效点，未绘制图形。\n');
end

%% ---------------------------------------------------------
% 11. 导出结果到 Excel
%% ---------------------------------------------------------
outFile = fullfile(pwd, 'duffing3_mlp_vs_true_results.xlsx');

% 时间历程 + 误差表（TimeSeries）- 增加速度数据
T = table( ...
    t(:), ...
    xTrue(:,1), xMLP(:,1), errX(:,1), ...
    vTrue(:,1), vMLP(:,1), vMLP(:,1) - vTrue(:,1), ...
    xTrue(:,2), xMLP(:,2), errX(:,2), ...
    vTrue(:,2), vMLP(:,2), vMLP(:,2) - vTrue(:,2), ...
    xTrue(:,3), xMLP(:,3), errX(:,3), ...
    vTrue(:,3), vMLP(:,3), vMLP(:,3) - vTrue(:,3), ...
    'VariableNames', { ...
        't', ...
        'x1_true','x1_mlp','x1_err', ...
        'v1_true','v1_mlp','v1_err', ...
        'x2_true','x2_mlp','x2_err', ...
        'v2_true','v2_mlp','v2_err', ...
        'x3_true','x3_mlp','x3_err', ...
        'v3_true','v3_mlp','v3_err' ...
    } ...
);

try
    writetable(T, outFile, 'Sheet', 'TimeSeries');
    fprintf('✅ 已导出时间历程数据（含位移和速度）到 TimeSeries 工作表\n');
catch ME
    warning('写入 TimeSeries 工作表失败: %s', ME.message);
end

% 参数信息表（Params）
paramCell = {
    'alpha',   alpha;
    'beta',    beta;
    'c',       c;
    'kc',      kc;
    'kappa',   kappa;
    'omegaP',  omegaP;
    'Fext',    Fext;
    'omegaF',  omegaF;
    'x1_0',    x1_0;
    'x2_0',    x2_0;
    'x3_0',    x3_0;
    'v1_0',    v1_0;
    'v2_0',    v2_0;
    'v3_0',    v3_0;
    'tEnd',    tEnd;
    'nSteps',  numel(t);
    'poincareOmega',  poincOmega;
    'poincarePeriod', poincT;
    'poincareTransient', poincTransient;
    'numPoincarePoints', numel(tPoinc);
    };

try
    writecell(paramCell, outFile, 'Sheet', 'Params', 'Range', 'A1');
catch ME
    warning('写入 Params 工作表失败: %s', ME.message);
end

% 庞加莱截面表（Poincare）：3 DOFs 全打进去
if ~isempty(tPoinc)
    TP = table( ...
        tPoinc(:), ...
        xTrueP(:,1), vTrueP(:,1), xMLPP(:,1), vMLPP(:,1), ...
        xTrueP(:,2), vTrueP(:,2), xMLPP(:,2), vMLPP(:,2), ...
        xTrueP(:,3), vTrueP(:,3), xMLPP(:,3), vMLPP(:,3), ...
        'VariableNames', { ...
            'tPoincare', ...
            'x1_true','v1_true','x1_mlp','v1_mlp', ...
            'x2_true','v2_true','x2_mlp','v2_mlp', ...
            'x3_true','v3_true','x3_mlp','v3_mlp' ...
        } ...
    );
    try
        writetable(TP, outFile, 'Sheet', 'Poincare');
    catch ME
        warning('写入 Poincare 工作表失败: %s', ME.message);
    end
end

fprintf('\n✅ 全部完成：MLP vs True（含外部强迫 + 3DOF 庞加莱截面），结果已导出到 Excel：\n   %s\n', outFile);


%% =========================================================
%  Local function 1: MLP ODE（外部强迫仅加在 dv1）
%% =========================================================
function dy = duffing3_mlp_ode(t,y,p)

    x1 = y(1); x2 = y(2); x3 = y(3);
    v1 = y(4); v2 = y(5); v3 = y(6);

    % 参数激励瞬时值: kappa_eff = kappa*cos(omegaP*t)
    kappa_eff = p.kappa * cos(p.omegaP * t);

    % 外部强迫项（不进 NN，只加在 dv1）
    F_eff = p.Fext * cos(p.omegaF * t);

    % 与训练数据一致的 7 维输入
    xInput = [x1 x2 x3 v1 v2 v3 kappa_eff];

    % NN 预测内部 ga1,ga2,ga3
    xNorm = p.normalizeInput(xInput);
    yNorm = predict(p.net, xNorm);
    acc   = p.denormalizeOutput(yNorm);
    acc   = double(acc(:));

    % 加速度（内部 + 外部强迫）
    dv1 = acc(1) + F_eff;
    dv2 = acc(2);
    dv3 = acc(3);

    dy = zeros(6,1);
    dy(1:3) = [v1; v2; v3];
    dy(4:6) = [dv1; dv2; dv3];
end


%% =========================================================
%  Local function 2: 真实 ODE（参数激励 + 外部强迫）
%% =========================================================
function dy = duffing3_true_ode(t,y,p)

    x1 = y(1); x2 = y(2); x3 = y(3);
    v1 = y(4); v2 = y(5); v3 = y(6);

    alpha  = p.alpha;
    beta   = p.beta;
    c      = p.c;
    kc     = p.kc;
    kappa  = p.kappa;
    omegaP = p.omegaP;
    Fext   = p.Fext;
    omegaF = p.omegaF;

    % 参数激励
    kappa_eff = kappa * cos(omegaP * t);
    % 外部强迫
    F_eff     = Fext  * cos(omegaF * t);

    dx1 = v1;
    dx2 = v2;
    dx3 = v3;

    dv1 = -c*v1 ...
          - (alpha + kappa_eff)*x1 ...
          - beta*x1^3 ...
          - kc*(x1 - x2) ...
          + F_eff;

    dv2 = -c*v2 - alpha*x2 - beta*x2^3 - kc*(2*x2 - x1 - x3);
    dv3 = -c*v3 - alpha*x3 - beta*x3^3 - kc*(x3 - x2);

    dy = [dx1; dx2; dx3; dv1; dv2; dv3];
end
