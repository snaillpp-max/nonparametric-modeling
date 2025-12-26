%% simulate_duffing3_mlp_from_wl.m 
% 用从 Mathematica 导入的 MLP + 原始 ODE 对比仿真 3DOF 参数激励 Duffing
% 状态: y = [x1,x2,x3,v1,v2,v3]^T
% MLP 预测: [ga1,ga2,ga3]   vs   原方程直接计算加速度

clear; clc; rng('shuffle');

fprintf('========== 3DOF 参数激励 Duffing + MLP 仿真（Mathematica 模型） ==========\n');

%% ---------------------------------------------------------
% 1. 加载从 Mathematica 导入的 MLP (ONNX + 归一化参数)
%% ---------------------------------------------------------
[net_pd3, inMean_pd3, inStd_pd3, tarMean_pd3, tarStd_pd3] = load_duffing3dof_nn();

% 为了兼容原脚本的命名，做一点变量映射（非必须，只是让下面写法更像你原来那份）
net     = net_pd3;
inMean  = inMean_pd3;
inStd   = inStd_pd3;
outMean = tarMean_pd3;
outStd  = tarStd_pd3;

% 防止标准差里出现 0
epsStd  = 1e-8;
inStd   = max(inStd, epsStd);
outStd  = max(outStd, epsStd);

normalizeInput    = @(x) (x - inMean) ./ inStd;
denormalizeOutput = @(yn) yn .* outStd + outMean;

fprintf('已加载 MLP 模型 (来自 Mathematica ONNX).\n');
fprintf('输入维度 = %d, 输出维度 = %d\n', numel(inMean), numel(outMean));

%% ---------------------------------------------------------
% 2. 物理参数（需要与你最初生成数据时一致）
%% ---------------------------------------------------------
alpha = -1;
beta  =  1;
c     =  0.3;
kc    =  1;

fprintf('物理参数: alpha=%g, beta=%g, c=%g, kc=%g\n', alpha,beta,c,kc);

%% ---------------------------------------------------------
% 3. 参数激励参数
%% ---------------------------------------------------------
kappa  = 1.2;       % 参数激励幅值
omegaP = 1;         % 参数激励频率 (rad/s)

T_p  = 2*pi/omegaP;
tEnd = 30*T_p;      % 仿真总时间，可以按需改长一点看长期行为

fprintf('参数激励: kappa=%g,  Omega_p=%g rad/s\n', kappa, omegaP);
fprintf('仿真时长 = %.2f 秒 ≈ %.1f 个参数周期\n', tEnd, tEnd/T_p);

%% ---------------------------------------------------------
% 4. 初值
%% ---------------------------------------------------------
x1_0 = 1;  x2_0 = 1;  x3_0 = 1;
v1_0 = 0;  v2_0 = 0;  v3_0 = 0;

y0 = [x1_0 x2_0 x3_0 v1_0 v2_0 v3_0]';

fprintf('初值:  x=[%.2f %.2f %.2f], v=[%.2f %.2f %.2f]\n', y0);

%% ---------------------------------------------------------
% 5. 参数结构体
%% ---------------------------------------------------------
paramMLP.net               = net;
paramMLP.normalizeInput    = normalizeInput;
paramMLP.denormalizeOutput = denormalizeOutput;
paramMLP.kappa             = kappa;
paramMLP.omegaP            = omegaP;

paramTrue.alpha  = alpha;
paramTrue.beta   = beta;
paramTrue.c      = c;
paramTrue.kc     = kc;
paramTrue.kappa  = kappa;
paramTrue.omegaP = omegaP;

%% ---------------------------------------------------------
% 6. 用相同时间网格，分别积分 MLP 版 & 真 ODE
%% ---------------------------------------------------------
opts  = odeset("RelTol",1e-6, "AbsTol",1e-8);
tSpan = linspace(0, tEnd, 4001);   % 公共输出时间点

fprintf('\n开始积分（MLP 模型）...\n');
[tMLP, yMLP] = ode45(@(t,y) duffing3_mlp_ode(t,y,paramMLP), tSpan, y0, opts);
fprintf('MLP 积分完成：%d 个时间点\n', numel(tMLP));

fprintf('\n开始积分（真实方程）...\n');
[tTrue, yTrue] = ode45(@(t,y) duffing3_true_ode(t,y,paramTrue), tSpan, y0, opts);
fprintf('真实 ODE 积分完成：%d 个时间点\n', numel(tTrue));

% tMLP 和 tTrue 理论上都是 tSpan，这里保险起见对齐一下
if any(abs(tMLP - tTrue) > 1e-10)
    warning('tMLP 与 tTrue 略有差异，使用插值对齐。');
    yTrue = interp1(tTrue, yTrue, tMLP);
    t    = tMLP;
else
    t    = tMLP;
end

xMLP  = yMLP(:,1:3); vMLP  = yMLP(:,4:6);
xTrue = yTrue(:,1:3); vTrue = yTrue(:,4:6);

%% ---------------------------------------------------------
% 7. 画对比图 & 误差
%% ---------------------------------------------------------
figure('Name','3DOF Duffing: MLP vs True (Mathematica 模型)','Position',[100 100 1100 800]);

% x1 时间历程对比
subplot(3,2,1);
plot(t, xTrue(:,1),'k','LineWidth',1.2); hold on;
plot(t, xMLP(:,1),'r--','LineWidth',1.0);
xlabel('t'); ylabel('x_1(t)');
legend('True','MLP');
title('x_1(t) 对比'); grid on;

% x1 误差
subplot(3,2,2);
plot(t, xMLP(:,1) - xTrue(:,1),'b','LineWidth',1.0);
xlabel('t'); ylabel('误差');
title('x_1(t) 误差 (MLP - True)'); grid on;

% x2 时间历程对比
subplot(3,2,3);
plot(t, xTrue(:,2),'k','LineWidth',1.2); hold on;
plot(t, xMLP(:,2),'r--','LineWidth',1.0);
xlabel('t'); ylabel('x_2(t)');
legend('True','MLP');
title('x_2(t) 对比'); grid on;

% x2 误差
subplot(3,2,4);
plot(t, xMLP(:,2) - xTrue(:,2),'b','LineWidth',1.0);
xlabel('t'); ylabel('误差');
title('x_2(t) 误差 (MLP - True)'); grid on;

% x3 时间历程对比
subplot(3,2,5);
plot(t, xTrue(:,3),'k','LineWidth',1.2); hold on;
plot(t, xMLP(:,3),'r--','LineWidth',1.0);
xlabel('t'); ylabel('x_3(t)');
legend('True','MLP');
title('x_3(t) 对比'); grid on;

% x3 误差
subplot(3,2,6);
plot(t, xMLP(:,3) - xTrue(:,3),'b','LineWidth',1.0);
xlabel('t'); ylabel('误差');
title('x_3(t) 误差 (MLP - True)'); grid on;

figure('Name','相图对比','Position',[200 100 900 700]);
subplot(2,2,1);
plot(xTrue(:,1), vTrue(:,1),'k','LineWidth',1.0); hold on;
plot(xMLP(:,1), vMLP(:,1),'r--','LineWidth',0.8);
xlabel('x_1'); ylabel('v_1');
legend('True','MLP'); title('相图 (x_1,v_1)'); grid on;

subplot(2,2,2);
plot(xTrue(:,2), vTrue(:,2),'k','LineWidth',1.0); hold on;
plot(xMLP(:,2), vMLP(:,2),'r--','LineWidth',0.8);
xlabel('x_2'); ylabel('v_2');
legend('True','MLP'); title('相图 (x_2,v_2)'); grid on;

subplot(2,2,3);
plot(xTrue(:,3), vTrue(:,3),'k','LineWidth',1.0); hold on;
plot(xMLP(:,3), vMLP(:,3),'r--','LineWidth',0.8);
xlabel('x_3'); ylabel('v_3');
legend('True','MLP'); title('相图 (x_3,v_3)'); grid on;

subplot(2,2,4);
plot3(xTrue(:,1), xTrue(:,2), xTrue(:,3),'k','LineWidth',1.0); hold on;
plot3(xMLP(:,1), xMLP(:,2), xMLP(:,3),'r--','LineWidth',0.8);
xlabel('x_1'); ylabel('x_2'); zlabel('x_3');
legend('True','MLP'); title('三维轨迹对比'); grid on;

fprintf('\n✅ 已完成 MLP (Mathematica 模型) 与原始 ODE 解的对比绘图。\n');


%% =========================================================
%  Local function 1: Duffing 3DOF + MLP 加速度预测 (调用导入的网络)
%% =========================================================
function dy = duffing3_mlp_ode(t,y,p)

    x1 = y(1); x2 = y(2); x3 = y(3);
    v1 = y(4); v2 = y(5); v3 = y(6);

    kappa_eff = p.kappa * cos(p.omegaP * t);

    % MLP 输入格式： [x1 x2 x3 v1 v2 v3 kappa_eff]
    xInput = [x1 x2 x3 v1 v2 v3 kappa_eff];

    xNorm = p.normalizeInput(xInput);
    yNorm = predict(p.net, xNorm);         % 输出是归一化的 [ga1 ga2 ga3]
    acc   = p.denormalizeOutput(yNorm);    % 反归一化到物理量
    acc   = double(acc(:));                % 3×1 列向量

    dy = zeros(6,1);
    dy(1:3) = [v1; v2; v3];
    dy(4:6) = acc;   % 这里就是 MLP 给出的 [ga1;ga2;ga3]
end


%% =========================================================
%  Local function 2: 真实 3DOF 参数 Duffing 右端
function dy = duffing3_true_ode(t,y,p)
    % 3DOF Duffing 真正 ODE（只对 DOF1 施加参数激励）
    %
    % x1'' + c x1' + (alpha + kappa_eff) x1 + beta x1^3 + kc (x1 - x2)       = 0
    % x2'' + c x2' + alpha x2            + beta x2^3       + kc (2x2-x1-x3)  = 0
    % x3'' + c x3' + alpha x3            + beta x3^3       + kc (x3 - x2)    = 0

    x1 = y(1); x2 = y(2); x3 = y(3);
    v1 = y(4); v2 = y(5); v3 = y(6);

    alpha  = p.alpha;
    beta   = p.beta;
    c      = p.c;
    kc     = p.kc;
    kappa  = p.kappa;
    omegaP = p.omegaP;

    kappa_eff = kappa * cos(omegaP * t);

    dx1 = v1;
    dx2 = v2;
    dx3 = v3;

    % 只在 DOF1 中使用 (alpha + kappa_eff)
    dv1 = -c*v1 ...
          - (alpha + kappa_eff)*x1 ...
          - beta*x1^3 ...
          - kc*(x1 - x2);

    % DOF2/3 使用常数 alpha（没有 kappa_eff）
    dv2 = -c*v2 ...
          - alpha*x2 ...
          - beta*x2^3 ...
          - kc*(2*x2 - x1 - x3);

    dv3 = -c*v3 ...
          - alpha*x3 ...
          - beta*x3^3 ...
          - kc*(x3 - x2);

    dy = [dx1; dx2; dx3; dv1; dv2; dv3];
end

