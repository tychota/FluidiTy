#import "@preview/cetz:0.5.2"
#let gray=rgb("edf1f3")
#let purple=rgb("eee2fa")
#let blue=rgb("dcefff")
#let orange=rgb("ffe8d2")
#let green=rgb("dbefe4")
#set page(paper:"a3",flipped:true,margin:(x:18mm,y:16mm),numbering:"1")
#set text(font:("Libertinus Serif",),size:10pt,lang:"en")
#set par(justify:false)
#let flow(draw)=align(center,cetz.canvas(length:1cm,{
 import cetz.draw: *
 set-style(stroke:.6pt,mark:(fill:rgb("344954")))
 for (x,label) in ((0,"PARTICLES · persistent state"),(12,"TRANSFERS · frozen support"),(24,"GRID · MAC faces and cells")) {
  rect((x - .2,1),(x + 10.2,-18.7),fill:luma(249),stroke:none)
  content((x+5,.65),text(font:"Libertinus Sans",weight:"bold",size:11pt,label))
 }
 let box(col,y,label,color,name)={
  let x=col*12
  rect((x,y),(x+10,y - 1.55),fill:color,radius:.08,name:name)
  content(name+".center",text(size:10pt,label))
 }
 let arr(a,b)=line(a,b,mark:(end:">"))
 let orth(a,b)=line(a,(a,"-|",b),b,mark:(end:">"))
 draw(box,arr,orth)
}))
#heading(numbering:none)[Detailed substep · ①–⑦: emission, activation, P2G, forces and contact]
Gray: shared operations; purple: MPM solids; blue: fluids; orange: gas; green: control. Each component $a$ has its own face lattice. All support positions are frozen at $x^n$; the accepted state is $A$, the trial state is $B$.
#flow((box,arr,orth)=>{
 import cetz.draw: *
 box(0,0,[Accepted particle state $A$\ $x_p,v_p,C_p,m_p,V_(0p)$; $J_p$ or $F$ or $(F_E,log J_"pl")$],gray,"state")
 box(0,-2.1,[① EMIT at frozen $Delta t$\ IDs = scan(parent id, ordinal); fuel and inflow sources],green,"emit")
 box(0,-4.2,[② SORT · stable (block, continuum)\ $B_d=floor((x_d/Delta x-1.5)/4)$],gray,"sort")
 box(1,-4.2,[② ACTIVATE and RESET · generation $g$\ Uniform: $\{B,B+1\}^3$; THB: certified global support\ $"acc"_m="acc"_q="acc"_f=0$ before every deposit],green,"activate")
 box(1,-6.4,[③ COMPATIBLE support $cal(C)_(p a)$\ $S=sum_(cal(C))N$; $N'=N/S$; $mu=sum_(cal(C))N'r$],gray,"support")
 box(0,-8.6,[③ CONSTITUTIVE law\ $tau_p="KirchhoffStress"("state"_p,S)$; Liquid/Gas: $tau=0$],purple,"stress")
 box(1,-8.6,[③ MASS and pure MOMENTUM · centred $r'=r-mu$\ $m_(i a)+=N'm_p$; $q_(i a)+=N'm_p(v_(p a)+c_(p a) dot r')$],gray,"mass")
 box(1,-10.8,[③ FORCE · true compatible gradient\ $f_(i a)+=-V_(0p)(tau_p nabla N'_(i a)(x_p))_a$],purple,"force")
 box(2,-9.2,[④ NORMALISE and snapshot before every force\ $u^0=q/m$; $v_"old"=u^0$; $m<m_min$: inactive],gray,"normal")
 box(2,-11.1,[⑤ BODY forces\ $u=u^0+Delta t(f/m+g_K+f_"fields")$; $g_K=0$ for Gas],gray,"body")
 box(2,-13.0,[⑤ THERMO · transport, reaction, heat\ $(dot(q),dot(p)_0,s_i,rho_"gas")$; $u+=Delta t b_"buoy"$],orange,"thermo")
 box(2,-14.9,[⑥ WAVES · Ocean Boundary and beach\ $u+= (1-e^(-Delta t/tau))(u_"target"-u)$],blue,"waves")
 box(2,-16.8,[⑦ MPM CONTACT · per-Body relative velocity\ $v_n<0$: $v_n^+=0$; $v_t^+=max(0,|v_t|-mu|v_n|)hat(t)$],purple,"contact")
 for (a,b) in (("state","emit"),("emit","sort"),("support","mass"),("mass","force"),("normal","body"),("body","thermo"),("thermo","waves"),("waves","contact")){arr(a+".south",b+".north")}
 arr("sort.east","activate.west")
 arr("activate.south","support.north")
 orth("sort.south","stress.north")
 orth("stress.east","force.west")
 arr("force.east","normal.west")
 content((5,-12.3),text(size:10pt)[*Deterministic P2G*\ $Q_c="rint"(c/s_c)$ for each particle contribution\ in all three channels $m,q,f$, before any sum.\ Integer i64 accumulation; checked range, no saturation.])
 content((5,-16.0),text(size:10pt)[*Support inclusion before writing*\ Half-cell displacement bound: $|Delta x_d|<=0.5Delta x$.\ Outside the shared tile: global i64 atomics if allocated;\ otherwise retry. Uniform tile only; THB uses\ certified global support. Fusion requires certification.])
 content((17,-14.1),text(size:10pt)[*Compatibility and affine fallback*\ $nabla N'=(nabla N-N'sum_(cal(C))nabla N)/S$.\ $D'=sum N'r'r'^T$; $S<0.1$ or singular $D'$:\ counted free flight, affine deposit disabled.\ Affine fit $C$ and deformation gradient $L$ are distinct.])
 content((29,-6.7),text(size:10pt)[*Explicit dependencies*\ Sort → activation → reset → every P2G write.\ Low-mass validity is decided only after deposition.\ In the narrow-band interior, retain Eulerian $u_E$.])
 content((29,-2.8),text(size:10pt)[*A shared grid, separate physical roles*\ One face-normal component per MAC face.\ Fluid viscosity acts on the grid at ⑨.\ Fluid no-penetration enters the pressure solve.\ ⑦ resolves MPM Body contact, using per-Body velocities.])
})
#text(size:9.5pt)[*Figure 11a.* Steps ①–⑦. The FLIP snapshot precedes application of stress forces, gravity, buoyancy, waves and contact. Compatible P2G uses centred moments; activation and accumulator reset are prerequisites of deposition. Disabled operators are identities.]
#pagebreak()
#heading(numbering:none)[Detailed substep · ⑧–⑩: geometry, dissipative viscosity and pressure]
Interface geometry is built before viscosity. Pressure is the last physical operation on grid velocity before G2P; extrapolated ghost values are then built for both the old snapshot and the new velocity.
#flow((box,arr,orth)=>{
 import cetz.draw: *
 box(2,0,[⑧ INTERFACE at $x^n$ · one plane per cell\ `sdf_sim` → fractions, apertures $a_f$, $theta,kappa,beta_f,J_f,s_"corr"$],blue,"interface")
 box(2,-2.3,[⑨ VISCOSITY · all fluid faces, band and interior\ $K_mu=E^T W_mu E$; $W_mu=2mu V$; $K_mu ≽ 0$],blue,"viscosity")
 box(2,-4.6,[⑨ EXPLICIT only with a certified spectral bound\ $Delta t Lambda_"cert"<=2(1-epsilon)$\ $u=u^star+Delta t M^(-1)(b_"wall"-K_mu u^star)$],blue,"explicit")
 box(2,-6.9,[⑨ otherwise IMPLICIT\ $(M+Delta t K_mu)u=M u^star+Delta t b_"wall"$],blue,"implicit")
 box(2,-9.2,[⑩ COMPATIBILITY · sealed incompressible components\ $sum V_i s_i=sum_"walls" (|f|-a_f)u_"obst" dot n$; fix one gauge],green,"compat")
 box(2,-11.5,[⑩ SPD SOLVE · integrated rows, MG-PCG\ $sum_f k_f(p_i-p_j)+h_i p_i=b_i$; $h_i=V_i/(K_i Delta t)$],blue,"solve")
 box(2,-13.8,[⑩ CORRECT · identical coefficients and jumps\ $u_f=u_f^star-(Delta t beta_f/d_f)(p_j-p_i-J_f)$],blue,"correct")
 box(1,-16.3,[⑩ → ⑪ GHOST extension on full frozen support\ $(v_"old",u) -> (hat(u)^"old",hat(u)^"new")$; apply blade-normal data],gray,"ghost")
 for (a,b) in (("interface","viscosity"),("viscosity","explicit"),("implicit","compat"),("compat","solve"),("solve","correct")){arr(a+".south",b+".north")}
 line("explicit.south","implicit.north",mark:(end:">"),stroke:(dash:"dashed"))
 line("explicit.east",(35,-5.375),(35,-9.975),"compat.east",mark:(end:">"))
 orth("correct.west","ghost.east")
 content((5,-2.1),text(size:10pt)[*Current volume → grid inertia*\ $V_p=J_p V_(0p)$; $rho_p=m_p/V_p$.\ Face inertia: $M_f=a_f d_f/beta_f$.\ The reference volume enters solid stress forces.])
 content((5,-6.0),text(size:10pt)[*A single viscous operator*\ Tensorial strain $E u$ includes cross derivatives.\ $u^T K_mu u >=0$: homogeneous boundaries dissipate\ in both branches under the explicit certificate.\ Moving walls add physical work through $b_"wall"$.\ Particle $C$ carries no viscous stress.])
 content((5,-10.4),text(size:10pt)[*Finite sound speed*\ $tilde(p)$ is physical pressure where $c<infinity$.\ $p^star="Advect"(tilde(p)^n)$ on all fluid cells.\ $K=rho_0 c_"ref"^2=rho c_"eff"^2$; commit $tilde(p)^(n+1)=p$ on acceptance.\ Compliance retains the mean: no gauge,\ no incompressible compatibility test there.])
 content((5,-15.1),text(size:10pt)[*Pressure validation in ⑬*\ $r=b-(cal(L)+H)p$; $H="diag"(h_i)$.\ $"div" u_i-s_i=-(p_i-p_i^star)/(K_i Delta t)-r_i/V_i$.\ Require this compliant residual to converge.\ At $c=infinity$, recover $"div" u=s$.])
 content((17,-2.2),text(size:10pt)[*Face conductance*\ $k_f=Delta t a_f beta_f/d_f$.\ Two phases: $beta_f=1/(theta rho_i+(1-theta)rho_j)$.\ $b_i=V_i s_i-Q_i^star-Q_("wall",i)+h_i p_i^star+"jumps"$.\ Free surface: $p_Gamma=p_"gas"+sigma kappa$, $J_f=0$\ at distance $max(theta,0.05)d_f$.])
 content((17,-6.6),text(size:10pt)[*Conservative geometry and work*\ Coarse flux = sum of fine physical fluxes.\ $G_"int"=-D^T$; pressure force $=-G_"int"p$.\ No volume division of matrix rows.\ The same $a_f,beta_f,d_f,J_f$ enter solve and correction.])
 content((17,-10.9),text(size:10pt)[*Explicit/implicit selection*\ $S=M^(-1/2)K_mu M^(-1/2)$.\ $Lambda_"cert"=max_i sum_j |S_(i j)| >= lambda_max(S)$.\ Cut cells, varying density/viscosity and THB\ are included in that bound.\ Without a certificate, use the implicit solve.])
 content((17,-14.0),text(size:10pt)[*Ghost construction*\ Incompatible or inactive faces use extrapolated data.\ Replace their blade-normal velocity by blade velocity,\ consistently for old and new fields.\ A constant boundary velocity gives zero FLIP increment.])
})
#text(size:9.5pt)[*Figure 11b.* Steps ⑧–⑩. The viscosity boxes denote alternative branches. Finite-$c$ cells retain their pressure as physical state and are validated against the compliant continuity equation. The pressure interface jump $J_f$ is distinct from particle volume ratio $J_p$.]
#pagebreak()
#heading(numbering:none)[Detailed substep · ⑪–⑭: full-stencil G2P, collision and acceptance]
G2P reads the full basis $N,nabla N$ at frozen $x^n$, including old and new ghost values. Each proposed write belongs to trial state $B$ until ⑬ accepts it; scheduled adaptation is a separate transaction afterward.
#flow((box,arr,orth)=>{
 import cetz.draw: *
 box(1,0,[⑪ PIC and FLIP · read before overwrite\ $v_("pic",a)=sum_i N hat(u)^"new"_(i a)$; $delta_a=v_(p a)-sum_i N hat(u)^"old"_(i a)$],gray,"pic")
 box(1,-2.2,[⑪ AFFINE FIT on full support\ $D^"full"_(p a)=sum_i N r r^T$; $c_(p a)=(D^"full"_(p a))^(-1)sum_i N hat(u)^"new"_(i a)r$],gray,"affine")
 box(1,-4.4,[⑪ DEFORMATION gradient · full true derivatives\ $L_p=sum_a e_a sum_i hat(u)^"new"_(i a)nabla N_(i a)(x_p)^T$],gray,"gradient")
 box(0,-4.4,[⑪ SOLID trial then return mapping\ $F_"trial"=(I+Delta t L_p)F$; $det F_"trial"<=0$: retry],purple,"trial")
 box(0,-6.6,[⑪ UPDATE constitutive state\ $(F,"hist")="ReturnMap"(F_"trial","hist",S)$; fluids: $J<-J exp(Delta t tr L)$],purple,"return")
 box(0,-8.8,[⑪ SEPARATION after return mapping\ $beta_p="Separation"("post-return state","contact") in {0,beta_max}$],gray,"separation")
 box(0,-11.0,[⑪ VELOCITY and ADVECTION contact\ $v_p=v_"pic"+alpha delta$; $v_"adv"=v_"pic"+beta_p alpha delta$; constrain both],purple,"vel")
 box(0,-13.2,[⑪ CONTINUOUS collision detection\ $x(t)=x^n+t v_"adv"$; conservative advancement / swept triangles],gray,"ccd")
 box(2,-13.2,[⑫ WHITEWATER events in trial $B$\ $(t,x,u,n,V_c Phi_c Delta t)$ per source cell and substep],blue,"events")
 box(2,-15.4,[⑬ VALIDATE and ACCEPT atomically\ finite + admissible + pressure residual; $A<-B$ only on ACCEPTED],green,"accept")
 box(1,-17.3,[⑭ ADAPT after ACCEPTED · separate transaction\ refine / coarsen / split / merge / reseed; commit generation $g+1$],green,"adapt")
 for (a,b) in (("pic","affine"),("affine","gradient"),("trial","return"),("return","separation"),("separation","vel"),("vel","ccd"),("events","accept")){arr(a+".south",b+".north")}
 arr("gradient.west","trial.east")
 line("ccd.east",(11,-13.975),(11,-12.7),(29,-12.7),"events.north",mark:(end:">"))
 orth("accept.west","adapt.east")
 content((5,-1.6),text(size:10pt)[*Two different gradients*\ $C_"affine"$ reconstructs local velocity.\ $L$ evolves deformation and drives separation.\ Do not substitute $C_"affine"$ for $L$.])
 content((17,-7.5),text(size:10pt)[*Affine fallback*\ Ill-conditioned $D^"full"$ → local PIC; set `kAffineOff`.\ Persist $C_p=L_p$ in this branch.\ Typed consumers read $C_"affine"=0$;\ deformation and separation still read the true $L$.])
 content((17,-11.2),text(size:10pt)[*Eulerian band transition*\ A particle without a valid old-grid history\ starts with $delta=0$, never a synthetic FLIP impulse.\ Reconstruct from the valid current full stencil.])
 content((17,-14.9),text(size:10pt)[*Collision accounting*\ Contact sets $beta=0$ and constrains both velocities.\ Find the first entry relative to the moving obstacle.\ Distance/motion bounds for conservative advancement,\ or continuous BVH swept-triangle intersection;\ final projection is a counted position correction.])
 content((29,-2.5),text(size:10pt)[*Constitutive updates*\ Elastic: identity return map.\ Granular / Snow: $F_E$ and $log J_"pl"$.\ Liquid / Gas: scalar $J$ only.\ Reference volume $V_0$ stays fixed during this substep.])
 content((29,-6.6),text(size:10pt)[*Ghost and cut bookkeeping*\ Full $N,nabla N,D^"full"$ in G2P,\ compatible $N',nabla N',D'$ in P2G.\ Exact force–deformation work identity away from cuts;\ record the cut-band defect and blade impulse.])
 content((29,-10.1),text(size:10pt)[*Rejection*\ RETRY_DT: discard $B$, restart from $A$ at $Delta t/2$.\ CAPACITY: leave graph; grow arenas; replay the frame.\ FATAL: stop with diagnostics.\ Discard rejected whitewater events and pressure history.])
})
#text(size:9.5pt)[*Figure 11c.* Steps ⑪–⑭. $alpha$ enters the FLIP contribution once. Contact constrains the actual advection velocity, and continuous collision detection prevents tunnelling. Whitewater events are committed with the accepted physical substep; adaptation cannot invalidate that accepted state.]
#pagebreak()
#set page(paper:"a4",flipped:false,margin:(x:25mm,y:22mm))
#set text(size:10.5pt)
#set par(justify:true)
