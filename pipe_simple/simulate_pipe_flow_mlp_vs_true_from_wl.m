%% =========================================================================
%   simulate_pipe_flow_mlp_vs_true_from_wl.m
%   Paidoussis 输流管（广义坐标 Nmodes=3）
%   对比：MLP（Mathematica ONNX） vs. 真 Galerkin ODE
%   初值：采用给定 U0 下的静态平衡解 q_eq(U0)，再加小速度扰动
%% =========================================================================
clear; clc; rng('shuffle');

fprintf('\n================== Paidoussis Pipe Flow: MLP vs TRUE (Static IC) ==================\n');

%% =========================================================================
%  1. 物理参数（与Mathematica 数据生成用的完全一致）
%% =========================================================================
L      = 1.0;
Douter = 0.01;
Dinner = 0.009;

EYoung = 2.06e11;
rhoS   = 7850;
rhoF   = 1000;

AreaPipe  = pi/4 * (Douter^2 - Dinner^2);
AreaFluid = pi/4 * Dinner^2;
Isec      = pi/64 * (Douter^4 - Dinner^4);

mLine = rhoS*AreaPipe + rhoF*AreaFluid;

T0    = 0.0;
cDamp = 0.3;

Nmodes = 3;

fprintf("材料参数/几何参数已加载，Nmodes = %d\n", Nmodes);

%% =========================================================================
%  2. Galerkin 模态形函数（简支）
%% =========================================================================
phi   = @(n,x) sin(n*pi*x/L);
dphi  = @(n,x) (n*pi/L)*cos(n*pi*x/L);
d2phi = @(n,x) -(n*pi/L)^2 * sin(n*pi*x/L);

% 数值积分
Nint = 2001;
xx = linspace(0,L,Nint);
dx = xx(2)-xx(1); %#ok<NASGU> % 纯占位

%% =========================================================================
%  3. 构造 Galerkin 矩阵（与 Mathematica 一致）
%% =========================================================================
Mmat   = zeros(Nmodes);
Kbend  = zeros(Nmodes);
KaxInt = zeros(Nmodes);
Gmat   = zeros(Nmodes);
Jmat   = zeros(Nmodes);

for i = 1:Nmodes
    for j = 1:Nmodes

        % 质量矩阵
        Mmat(i,j) = mLine * trapz(xx, phi(i,xx).*phi(j,xx));

        % 弯曲刚度
        Kbend(i,j) = EYoung*Isec * trapz(xx, d2phi(i,xx).*d2phi(j,xx));

        % 轴向积分矩阵
        KaxInt(i,j) = trapz(xx, dphi(i,xx).*dphi(j,xx));

        % G 矩阵
        Gmat(i,j) = trapz(xx, phi(i,xx).*dphi(j,xx));

        % J 矩阵
        Jmat(i,j) = trapz(xx, phi(i,xx).*d2phi(j,xx));
    end
end

% 阻尼矩阵 C = (cDamp/mLine) M
Cmat = (cDamp/mLine) * Mmat;

fprintf("Galerkin 矩阵 M, C, Kbend, KaxInt, G, J 构造完成。\n");
%% =========================================================================
%  3.5 均布外激励：f(x,t) = f0 * cos(OmegaF*t)（单位：N/m，沿长度均布）
%% =========================================================================
f0     = 0.00;            % 均布载荷幅值（N/m），0表示不加外激励
OmegaF = 1*13.2984*2*pi;      % 外激励角频率（rad/s）
U0    = 80;     % 平均流速（建议在 U0List 里取，比如 70~80）
dU    = 1.3;      % 扰动幅值
Omega = 0.9*13.2984*2*pi;    % 扰动频率（可以取对应 U0 的 f1 倍数）

% Galerkin 广义力：F_n(t) = ∫_0^L phi_n(x) * f(x,t) dx
% 对均布：f(x,t)=f0*cos(OmegaF*t) => F_n(t) = ( ∫ phi_n dx ) * f0*cos(...)
bForce = zeros(Nmodes,1);
for n = 1:Nmodes
    bForce(n) = trapz(xx, phi(n,xx));   % 等价于 ∫_0^L sin(nπx/L) dx
end

fprintf("外激励: f0=%g (N/m), OmegaF=%g rad/s\n", f0, OmegaF);

%% =========================================================================
%  4. 加载 MLP（ONNX）和归一化参数
%% =========================================================================
onnxFile = "trainedNet_pipe_flow_cos_final.onnx";
matFile  = "normParams_pipe_flow_cos_final.mat";

fprintf("加载 ONNX: %s\n", onnxFile);

% ONNX 输入 shape = (1, 8) → 数据格式用 "BC" (Batch, Channel)
net = importNetworkFromONNX(onnxFile, ...
    "InputDataFormats","BC");
fprintf("加载归一化参数: %s\n", matFile);
S = load(matFile);

% 一些版本里是直接保存为 4 个变量：inputMean,inputStd,targetMean,targetStd
% 一些版本里是保存成一个 struct，比如 normParams.inputMean 等
if isfield(S, "inputMean") && isfield(S,"inputStd") ...
        && isfield(S,"targetMean") && isfield(S,"targetStd")

    % 顶层就有四个字段：最简单的情况
    inMean  = S.inputMean(:).';
    inStd   = S.inputStd(:).';
    outMean = S.targetMean(:).';
    outStd  = S.targetStd(:).';

else
    % 顶层不是这四个字段，而是包了一层，比如 S.normParams.XXX
    fns = fieldnames(S);
    if numel(fns) ~= 1
        error("在 %s 里找不到 inputMean 等字段，顶层字段有：%s", ...
              matFile, strjoin(fns.', ", "));
    end
    P = S.(fns{1});  % 例如 P = S.normParams

    if ~all(isfield(P, ["inputMean","inputStd","targetMean","targetStd"]))
        error("文件 %s 顶层唯一字段 '%s' 里也没有 inputMean / inputStd / targetMean / targetStd。", ...
              matFile, fns{1});
    end

    inMean  = P.inputMean(:).';
    inStd   = P.inputStd(:).';
    outMean = P.targetMean(:).';
    outStd  = P.targetStd(:).';
end


epsStd = 1e-8;
inStd  = max(inStd,epsStd);
outStd = max(outStd,epsStd);


normalizeInput    = @(x) (x - inMean)./inStd;
denormalizeOutput = @(y) y.*outStd + outMean;

fprintf("MLP 模型加载完成。输入维=%d 输出维=%d\n", numel(inMean), numel(outMean));

%% =========================================================================
%  5. 选择流动参数 (U0, dU, Omega)
%% =========================================================================


fprintf("流动参数: U0=%g, dU=%g, Omega=%g\n", U0, dU, Omega);

T_p  = 2*pi/max(Omega,1e-6);
tEnd = 300*T_p;

%% =========================================================================
%  6. 静态平衡解 q_eq(U0) 作为初值
%     解非线性方程：Klin(U0)*q - (EAp/(2L))*Sgeo(q)*J*q = 0
%% =========================================================================
fprintf("\n求解静态平衡解 q_eq(U0=%.3g)...\n", U0);
q_eq = static_equilibrium_pipe(U0, Kbend, KaxInt, Jmat, ...
                               EYoung, AreaPipe, AreaFluid, rhoF, L, T0, Nmodes, phi);

fprintf("静态平衡解范数 ||q_eq|| = %.3e\n", norm(q_eq));


q0    =q_eq ;
%q0    =[0.004451937,-4.1168E-05,-2.38274E-06]';


qdot0 =0.1* ones(Nmodes,1);
%qdot0 =[0.059647125,0.011290532,-0.000216766]';

y0 = [q0; qdot0];
 
fprintf("初值采用静态平衡 + 小速度扰动：\n");
qdisp   = zeros(1,3);  qdisp(1:min(3,Nmodes))   = q0(1:min(3,Nmodes));
qdotdisp= zeros(1,3);  qdotdisp(1:min(3,Nmodes))= qdot0(1:min(3,Nmodes));
fprintf("  q0   = [%.3e %.3e %.3e]\n",   qdisp(1),   qdisp(2),   qdisp(3));
fprintf("  qdot0= [%.3e %.3e %.3e]\n", qdotdisp(1), qdotdisp(2), qdotdisp(3));


%% =========================================================================
%  7. 积分：MLP 模型 vs. 真 ODE
%% =========================================================================
opts  = odeset("RelTol",1e-4,"AbsTol",1e-6);
tSpan = linspace(0,tEnd,20001);

fprintf("\n开始积分（MLP）...\n");
[tMLP, yMLP] = ode45(@(t,y) pipe_mlp_ode(t,y,net,normalizeInput,denormalizeOutput, ...
                                         U0,dU,Omega,Nmodes, ...
                                         Mmat,f0,OmegaF,bForce), ...
                     tSpan, y0, opts);

fprintf("开始积分（TRUE ODE）...\n");
[tTrue, yTrue] = ode45(@(t,y) pipe_true_ode(t,y,Mmat,Cmat,Kbend,KaxInt,Gmat,Jmat, ...
                                            EYoung,AreaPipe,AreaFluid,rhoF,L,T0, ...
                                            U0,dU,Omega,Nmodes, ...
                                            f0,OmegaF,bForce), ...
                       tSpan, y0, opts);


t = tMLP;  % 理论上一致

qMLP     = yMLP(:,1:Nmodes);
qTrue    = yTrue(:,1:Nmodes);
qdotMLP  = yMLP(:,Nmodes+1:end);
qdotTrue = yTrue(:,Nmodes+1:end);

%% =========================================================================
%  8. 可视化：模态位移 & 相图
%% =========================================================================
t = tMLP;  % 理论上一致

qMLP     = yMLP(:,1:Nmodes);
qTrue    = yTrue(:,1:Nmodes);
qdotMLP  = yMLP(:,Nmodes+1:end);
qdotTrue = yTrue(:,Nmodes+1:end);

%% =========================================================================
%  8. 可视化：q1–q3 + w(L/2,t)（Duffing 风格）+ 模态相图 + 总位移相图
%% =========================================================================

nPlot = min(3, Nmodes);   % 最多画前三个模态

% ---------- 8.1 中点挠度 w(L/2,t) ----------
xMid = L/2;

phiMid = zeros(Nmodes,1);
for n = 1:Nmodes
    phiMid(n) = phi(n, xMid);
end

% w(t) = Σ q_n(t) phi_n
wTrue = qTrue * phiMid;
wMLP  = qMLP  * phiMid;

% ẇ(t) = Σ q̇_n(t) phi_n
wdotTrue = qdotTrue * phiMid;
wdotMLP  = qdotMLP  * phiMid;

%% ---------- 图1：q1–q3 + w(L/2,t) ----------
figure('Name','q1,q2,q3 & w(L/2,t) True vs MLP', ...
       'Position',[100 100 1000 600]);

% 上图：q1,q2,q3
subplot(2,1,1); hold on;

plot(t, qTrue(:,1), 'k-', 'LineWidth',1.2);
plot(t, qMLP(:,1),  'r--','LineWidth',1.0);

if Nmodes >= 2
    plot(t, qTrue(:,2), 'b-', 'LineWidth',1.2);
    plot(t, qMLP(:,2),  'c--','LineWidth',1.0);
end

if Nmodes >= 3
    plot(t, qTrue(:,3), 'm-', 'LineWidth',1.2);
    plot(t, qMLP(:,3),  'g--','LineWidth',1.0);
end

xlabel('t (s)'); ylabel('q_n(t)');
legend({'q_1 True','q_1 MLP', ...
        'q_2 True','q_2 MLP', ...
        'q_3 True','q_3 MLP'}, 'Location','best');
title('广义坐标 q_1,q_2,q_3 时间历程（True vs MLP）');
grid on;

% 下图：w(L/2)
subplot(2,1,2); hold on;
plot(t, wTrue, 'k','LineWidth',1.2);
plot(t, wMLP , 'r--','LineWidth',1.0);
xlabel('t (s)'); ylabel('w(L/2,t)');
legend('True','MLP','Location','best');
title('中点挠度 w(L/2,t)（True vs MLP）');
grid on;


%% =========================================================================
%  9. 导出 Excel：时间历程 & 庞加莱截面数据（位移+速度）
%% =========================================================================

% ---------- 9.1 时间历程导出（含误差） ----------

% 位移误差：MLP - True
qErr    = qMLP - qTrue;
qdotErr = qdotMLP - qdotTrue;

% 中点位移误差
wErr    = wMLP - wTrue;
wdotErr = wdotMLP - wdotTrue;

% 绝对误差
qAbsErr    = abs(qErr);
qdotAbsErr = abs(qdotErr);
wAbsErr    = abs(wErr);
wdotAbsErr = abs(wdotErr);

% 相对误差，避免除以零
epsRel = 1e-12;
qRelErr    = abs(qErr)    ./ max(abs(qTrue),    epsRel);
qdotRelErr = abs(qdotErr) ./ max(abs(qdotTrue), epsRel);
wRelErr    = abs(wErr)    ./ max(abs(wTrue),    epsRel);
wdotRelErr = abs(wdotErr) ./ max(abs(wdotTrue), epsRel);

% 百分比相对误差
qRelErrPct    = 100 * qRelErr;
qdotRelErrPct = 100 * qdotRelErr;
wRelErrPct    = 100 * wRelErr;
wdotRelErrPct = 100 * wdotRelErr;

timeHeader = { ...
    't', ...
    ...
    'q1True','q1dotTrue','q1MLP','q1dotMLP','q1Err','q1dotErr','q1AbsErr','q1dotAbsErr','q1RelErrPct','q1dotRelErrPct', ...
    'q2True','q2dotTrue','q2MLP','q2dotMLP','q2Err','q2dotErr','q2AbsErr','q2dotAbsErr','q2RelErrPct','q2dotRelErrPct', ...
    'q3True','q3dotTrue','q3MLP','q3dotMLP','q3Err','q3dotErr','q3AbsErr','q3dotAbsErr','q3RelErrPct','q3dotRelErrPct', ...
    ...
    'wTrue','wdotTrue','wMLP','wdotMLP','wErr','wdotErr','wAbsErr','wdotAbsErr','wRelErrPct','wdotRelErrPct'};

timeData = [ ...
    t(:), ...
    ...
    qTrue(:,1), qdotTrue(:,1), qMLP(:,1), qdotMLP(:,1), qErr(:,1), qdotErr(:,1), qAbsErr(:,1), qdotAbsErr(:,1), qRelErrPct(:,1), qdotRelErrPct(:,1), ...
    qTrue(:,2), qdotTrue(:,2), qMLP(:,2), qdotMLP(:,2), qErr(:,2), qdotErr(:,2), qAbsErr(:,2), qdotAbsErr(:,2), qRelErrPct(:,2), qdotRelErrPct(:,2), ...
    qTrue(:,3), qdotTrue(:,3), qMLP(:,3), qdotMLP(:,3), qErr(:,3), qdotErr(:,3), qAbsErr(:,3), qdotAbsErr(:,3), qRelErrPct(:,3), qdotRelErrPct(:,3), ...
    ...
    wTrue(:), wdotTrue(:), wMLP(:), wdotMLP(:), wErr(:), wdotErr(:), wAbsErr(:), wdotAbsErr(:), wRelErrPct(:), wdotRelErrPct(:)];

timeFile = sprintf('pipe_time_series_with_error_U0_%g_dU_%g_Omega_%g.xlsx', U0, dU, Omega);
writecell([timeHeader; num2cell(timeData)], timeFile);
fprintf('✅ 时间历程及误差数据已导出到 %s\n', timeFile); 

% ---------- 9.2 计算庞加莱截面（以激励周期 T = 2π/Omega 为截面） ----------
T   = 2*pi/Omega;        % 激励周期
dt  = t(2) - t(1);       % 采样步长（t 是 linspace 出来的）
t0  = t(1);
tEndEff = t(end);

nPeriod = floor((tEndEff - t0)/T);    % 总周期数

if nPeriod < 5
    warning('有效周期数太少（%d），庞加莱截面点会比较少。', nPeriod);
end

% 跳过前 20%% 周期作为过渡
nSkip = max(1, floor(0.5 * nPeriod));
kList = (nSkip:nPeriod).';

tPoincare = t0 + kList*T;                    % 理想截面时间
idxP = round((tPoincare - t0)/dt) + 1;       % 对应索引（t 是等间距）

% 保证不越界 & 唯一
idxP = idxP(idxP >= 1 & idxP <= numel(t));
idxP = unique(idxP);

tP      = t(idxP);
q1TrueP = qTrue(idxP,1);
q1dotTrueP = qdotTrue(idxP,1);
q1MLPP  = qMLP(idxP,1);
q1dotMLPP = qdotMLP(idxP,1);

wTrueP   = wTrue(idxP);
wdotTrueP= wdotTrue(idxP);
wMLPP    = wMLP(idxP);
wdotMLPP = wdotMLP(idxP);

% ---------- 9.3 庞加莱截面导出 ----------
poincareHeader = { ...
    'tP', ...
    'q1True','q1dotTrue','q1MLP','q1dotMLP', ...
    'q2True','q2dotTrue','q2MLP','q2dotMLP', ...
    'q3True','q3dotTrue','q3MLP','q3dotMLP', ...
    'wTrue','wdotTrue','wMLP','wdotMLP'};


poincareData = [ ...
    tP(:), ...
    qTrue(idxP,1), qdotTrue(idxP,1), qMLP(idxP,1), qdotMLP(idxP,1), ...
    qTrue(idxP,2), qdotTrue(idxP,2), qMLP(idxP,2), qdotMLP(idxP,2), ...
    qTrue(idxP,3), qdotTrue(idxP,3), qMLP(idxP,3), qdotMLP(idxP,3), ...
    wTrueP(:), wdotTrueP(:), ...
    wMLPP(:),  wdotMLPP(:)];


poincareFile = sprintf('pipe_poincare_U0_%g_dU_%g_Omega_%g.xlsx', U0, dU, Omega);
writecell([poincareHeader; num2cell(poincareData)], poincareFile);
fprintf('✅ 庞加莱截面数据已导出到 %s\n', poincareFile);

% ---------- 9.4 可选：画庞加莱截面图（总位移 w-wdot） ----------


for k = 1:nPlot
    subplot(nPlot,2,2*k-1);
    plot(t,qTrue(:,k),'k','LineWidth',1.2); hold on;
    plot(t,qMLP(:,k),'r--','LineWidth',1.0);
    xlabel("t"); ylabel(sprintf("q_%d",k));
    legend("True","MLP");
    title(sprintf("模态 %d 位移对比",k));
    grid on;

    subplot(nPlot,2,2*k);
    plot(t,qMLP(:,k)-qTrue(:,k),'b');
    xlabel("t"); ylabel("误差");
    title(sprintf("q_%d 误差",k));
    grid on;
end



%% =========================================================================
%  函数：静态平衡 (静分岔点) 求解
%  解: Klin(U0)*q - (EAp/(2L))*Sgeo(q)*J*q = 0
%% =========================================================================
function q_eq = static_equilibrium_pipe(U0,Kbend,KaxInt,Jmat, ...
                                        EYoung,AreaPipe,AreaFluid,rhoF,L,T0,Nmodes,phi)

    % 线性刚度（静态问题与动力学一致，只是没有 U(t) 变化）
    Klin = Kbend + (T0 - rhoF*AreaFluid*U0^2) * KaxInt;

    % 非线性平衡方程残差
    function F = staticResidual(q)
        q    = q(:);
        Sgeo = q.' * KaxInt * q;                  % 标量
        geoVec = -(EYoung*AreaPipe/(2*L)) * Sgeo * (Jmat*q);  % N×1
        F = Klin*q + geoVec;   % 对应 eqsStatic = 0
    end

    opts = optimoptions('fsolve', ...
        'Display','off', ...
        'FunctionTolerance',1e-12, ...
        'StepTolerance',1e-12, ...
        'MaxIterations',800, ...
        'MaxFunctionEvaluations',5000);

    % 多个初值尝试：零解 + 小一阶模态 + 小随机扰动
    guesses =0.01*[1; zeros(Nmodes-1,1)];

    best_q = zeros(Nmodes,1);
    best_res = inf;

    for k = 1:size(guesses,2)
        q0_try = guesses(:,k);
        try
            q_sol = fsolve(@staticResidual,q0_try,opts);
            res_norm = norm(staticResidual(q_sol));
            if res_norm < best_res
                best_res = res_norm;
                best_q   = q_sol;
            end
        catch
            % ignore failure, try next guess
        end
    end

    % 评估中跨挠度（和 Mathematica 的筛选逻辑类似）
    xMid = L/2;
    wMid = 0;
    for n = 1:Nmodes
        wMid = wMid + best_q(n)*phi(n,xMid);
    end

    fprintf("  静态平衡求解完成：残差范数=%.3e, 中跨挠度 w(L/2)=%.3e\n", ...
            best_res, wMid);

    % 如果残差太大，就退回全零（至少是一个线性平衡解）
    if best_res > 1e-6
        warning("静态平衡残差较大，退回 q_eq = 0.");
        q_eq = zeros(Nmodes,1);
    else
        q_eq = best_q;
    end
end

%% =========================================================================
%  函数：MLP 版 ODE
%% =========================================================================
function dy = pipe_mlp_ode(t,y,net,normalizeInput,denormalizeOutput,U0,dU,Omega,N, ...
                           Mmat,f0,OmegaF,bForce)

    q    = y(1:N).';
    qdot = y(N+1:2*N).';

    U_t    = U0;
    dUcos  = dU*cos(Omega*t);

    % MLP 输入格式: [U0, dU*cos(Ω t), q1..qN, q1dot..qNdot]
    xInput = [U_t, dUcos, q, qdot];

    xNorm = normalizeInput(xInput);
    yNorm = predict(net,xNorm);               % 1×N
    acc_internal = denormalizeOutput(yNorm);  % 1×N

    % ===== 外激励（Galerkin 广义力）=====
    % F(t) = bForce * f0*cos(OmegaF*t)
    if f0 ~= 0 && OmegaF ~= 0
        Fvec = bForce(:) * (f0*cos(OmegaF*t));  % N×1
        a_forcing = Mmat \ Fvec;                % N×1
    else
        a_forcing = zeros(N,1);
    end

    acc_total = acc_internal(:) + a_forcing;

    dy = zeros(2*N,1);
    dy(1:N)     = qdot(:);
    dy(N+1:2*N) = acc_total(:);
end

%% =========================================================================
%  函数：真实 Paidoussis ODE
%% =========================================================================
function dy = pipe_true_ode(t,y,Mmat,Cmat,Kbend,KaxInt,Gmat,Jmat, ...
                            EYoung,AreaPipe,AreaFluid,rhoF,L,T0, ...
                            U0,dU,Omega,N, ...
                            f0,OmegaF,bForce)

    q    = y(1:N);
    qdot = y(N+1:2*N);

    U = U0 + dU*cos(Omega*t);

    Klin = Kbend + (T0 - rhoF*AreaFluid*U^2) * KaxInt;

    Sgeo   = q.' * KaxInt * q;
    geoVec = -(EYoung*AreaPipe/(2*L)) * Sgeo * (Jmat*q);

    Ceff = Cmat + 2*rhoF*AreaFluid*U*Gmat;

    % ===== 外激励（Galerkin 广义力）=====
    if f0 ~= 0 && OmegaF ~= 0
        Fvec = bForce(:) * (f0*cos(OmegaF*t));   % N×1
    else
        Fvec = zeros(N,1);
    end

    % M q'' + Ceff q' + Klin q + geoVec = Fvec
    rhs   = -(Ceff*qdot + Klin*q + geoVec) + Fvec;
    qddot = Mmat \ rhs;

    dy = [qdot; qddot];
end
