#import "@preview/cetz:0.5.2"
#let ink=rgb("254657")
#let pale=rgb("e4eff5")
#let warm=rgb("f5e8d9")
#let lilac=rgb("eee5f3")
#let grid-figure()=cetz.canvas(length:1cm,{
 import cetz.draw: *
 set-style(stroke:.6pt,mark:(fill:ink))
 // Left: one staggered MAC grid shared by every continuum
 content((2,4.75),text(font:"Libertinus Sans",weight:"bold",size:10pt,"One staggered MAC grid"))
 for k in range(5){line((k,0),(k,4),stroke:luma(190));line((0,k),(4,k),stroke:luma(190))}
 for x in range(4){for y in range(4){circle((x+.5,y+.5),radius:.065,fill:rgb("a76829"),stroke:none)}}
 for x in range(5){for y in range(4){line((x - .15,y+.5),(x+.15,y+.5),stroke:1pt+ink,mark:(end:">"))}}
 for x in range(4){for y in range(5){line((x+.5,y - .15),(x+.5,y+.15),stroke:1pt+rgb("377b68"),mark:(end:">"))}}
 for (x,y,c) in ((1.3,1.6,rgb("9572ae")),(2.7,2.25,rgb("3d7fb0")),(.7,2.8,rgb("9572ae")),(3.2,.8,rgb("3d7fb0"))){circle((x,y),radius:.11,fill:c)}
 content((2,-.55),text(size:9pt)[Cells: $p, phi$, fractions · faces: $u_a$])
 content((2,-1.05),text(size:9pt)[Particles of every continuum deposit per face])
 // Right: dyadic nesting in 1D
 let ox=6.2
 content((ox+3,4.75),text(font:"Libertinus Sans",weight:"bold",size:10pt,"Why faces and cells nest (1D)"))
 // cell-centred quadratic
 line((ox,3.2),(ox+6,3.2),stroke:luma(150))
 for k in range(4){line((ox+2*k,3.05),(ox+2*k,3.35),stroke:luma(120))}
 circle((ox+3,3.2),radius:.08,fill:rgb("a76829"),stroke:none)
 content((ox+3,3.65),text(size:8.5pt)[coarse cell centre, quadratic])
 line((ox,2.2),(ox+6,2.2),stroke:luma(150))
 for k in range(7){line((ox+k,2.1),(ox+k,2.3),stroke:luma(160))}
 for (x,w) in ((1.5,"1/4"),(2.5,"3/4"),(3.5,"3/4"),(4.5,"1/4")){circle((ox+x,2.2),radius:.06,fill:rgb("a76829"),stroke:none);content((ox+x,1.9),text(size:8pt,w));line((ox+3,3.12),(ox+x,2.28),stroke:(paint:luma(150),dash:"dashed"))}
 // face-centred cubic
 line((ox,1.0),(ox+6,1.0),stroke:luma(150))
 for k in range(4){line((ox+2*k,.85),(ox+2*k,1.15),stroke:luma(120))}
 line((ox+2 - .2,1.0),(ox+2+.2,1.0),stroke:1pt+ink,mark:(end:">"))
 content((ox+4.2,1.35),text(size:8.5pt)[coarse face, cubic])
 line((ox,0),(ox+6,0),stroke:luma(150))
 for k in range(7){line((ox+k,-.1),(ox+k,.1),stroke:luma(160))}
 for (x,w) in ((0,"1/8"),(1,"4/8"),(2,"6/8"),(3,"4/8"),(4,"1/8")){line((ox+x - .15,0),(ox+x+.15,0),stroke:1pt+ink,mark:(end:">"));content((ox+x,-.35),text(size:8pt,w));line((ox+2,.92),(ox+x,.08),stroke:(paint:luma(150),dash:"dashed"))}
 content((ox+3,-1.05),text(size:9pt)[Masks $(1,3,3,1)\/4$ and $(1,4,6,4,1)\/8$])
})
#let stencil-figure()=cetz.canvas(length:1cm,{
 import cetz.draw: *
 set-style(stroke:.6pt,mark:(fill:ink))
 for x in range(5){line((x,0),(x,3),stroke:luma(210))}
 for y in range(4){line((0,y),(4,y),stroke:luma(210))}
 rect((.5,0),(3.5,3),fill:pale.transparentize(40%),stroke:none)
 for x in range(3){for y in range(3){
  let px=x+1; let py=y+.5
  line((px - .14,py),(px+.14,py),stroke:1pt+ink,mark:(end:">"))
  line((1.6,1.3),(px,py),stroke:(paint:ink.lighten(45%),dash:"dashed"))}}
 circle((1.6,1.3),radius:.12,fill:rgb("a66a38"))
 content((1.6,.95),text(size:9pt)[$x_p$])
 content((2,3.5),text(size:10pt)[$x$-faces of one particle: 3 × 3 samples])
 rect((5,0),(12.4,3.1),radius:.1,fill:lilac)
 content((8.7,2.65),text(size:10pt,weight:"bold")[A particle carries a local affine field])
 content((8.7,2.0),text(size:10pt)[$v_p (x)=v_p+C_p (x-x_p)$, $C_p$ stored])
 content((8.7,1.25),text(size:10pt)[$D^"full"_(p a)=sum_i N_(i p a) r_(i p a) r_(i p a)^T$])
 content((8.7,.5),text(size:10pt)[$c_(p a)=(D^"full"_(p a))^(-1) sum_i N_(i p a) hat(u)_(i a) r_(i p a)$])
 line((3.9,1.5),(4.8,1.5),mark:(end:">"))
 content((6.2,-.65),text(size:9pt)[Full-stencil G2P includes ghost faces; $C_"affine"$ is a fit, while $L=sum hat(u) nabla N^T$ evolves deformation.])
})
#let stress-figure()=cetz.canvas(length:1cm,{
 import cetz.draw: *
 set-style(stroke:.7pt,mark:(fill:ink))
 rect((0,0),(2,2),fill:pale)
 line((2.5,1),(4,1),mark:(end:">"));content((3.25,1.4),[$F$])
 line((4.5,0),(7,.3),(7.5,2.2),(5,1.9),(4.5,0),fill:warm)
 line((6.3,1.25),(7.5,1.25),mark:(end:">"));content((7.4,1.65),[$sigma_C n$])
 content((1,-.55),text(size:9pt)[Reference volume $V_0$])
 content((6,-.55),text(size:9pt)[Current volume $V=J V_0$])
 rect((8.5,-.1),(13,2.2),fill:lilac,radius:.1)
 content((10.75,1.7),text(size:10pt)[$J=det F$])
 content((10.75,1),text(size:10pt)[$P=partial Psi\/partial F$])
 content((10.75,.3),text(size:10pt)[$tau=P F^T=J sigma_C$])
 content((6.5,-1.25),text(size:10pt)[$f_(i a)=-sum_p V_(0p)(tau_p nabla N'_(i a)(x_p))_a$; outside cuts $N'=N$ and force–deformation work is exact.])
})
#let interface-figure()=cetz.canvas(length:1cm,{
 import cetz.draw: *
 set-style(stroke:.65pt,mark:(fill:ink))
 rect((0,0),(2,2),fill:pale);rect((2,0),(4,2),fill:warm)
 line((2.6,0),(2.6,2),stroke:1.5pt+ink)
 circle((1,1),radius:.07,fill:ink);circle((3,1),radius:.07,fill:ink)
 line((1,1),(2.6,1),mark:(end:">"));content((1.8,1.4),text(size:9pt)[$theta d_f$])
 content((1,.4),[$p_i$]);content((3,.4),[$p_j$])
 content((2,2.5),text(size:10pt)[Interface crossing a pressure link])
 content((2,-.5),text(size:9pt)[$beta_f=1\/(theta rho_i+(1-theta)rho_j)$])
 for (x,y,w,h) in ((7,0,2,2),(9,0,1,1),(9,1,1,1),(10,0,1,1),(10,1,1,1)){rect((x,y),(x+w,y+h),fill:pale,stroke:.7pt)}
 line((8.5,.5),(9.5,.5),mark:(end:">"));line((8.5,1.5),(9.5,1.5),mark:(end:">"))
 content((9,2.5),text(size:10pt)[2:1 coarse–fine flux interface])
 content((9,-.5),text(size:10pt)[$Q_"coarse"=sum_(f in "children")Q_f$])
})
#let mesh-figure()=cetz.canvas(length:1cm,{
 import cetz.draw: *
 set-style(stroke:.65pt,mark:(fill:ink))
 let box(x,label,col,name)={rect((x,0),(x+3.5,1.5),fill:col,radius:.1,name:name);content(name+".center",block(width:3.3cm,align(center,text(size:9pt,label))))}
 box(0,[Simulate (GPU)\ `sdf_sim`, θ, κ, pressure\ band particles, velocity],pale,"sim")
 box(4.6,[Mesh input field\ classical baseline\ optional neural GPU],lilac,"sdf")
 box(9.2,[Mostly CPU tracker\ advection + projection\ repair and export],warm,"mesh")
 line("sim.east","sdf.west",mark:(end:">"));line("sdf.east","mesh.west",mark:(end:">"))
 content((6.35,-.7),text(size:9pt)[Data flow is one-way: Simulate never reads `mesh/`; rerunning Mesh leaves `sim/` unchanged.])
})
#let bins-figure()=cetz.canvas(length:1cm,{
 import cetz.draw: *
 set-style(stroke:.6pt,mark:(fill:ink))
 content((2.3,3.7),text(size:10pt,weight:"bold")[Half-offset particle bins → shared tile])
 for row in range(3){
  content((.1,2.8-row*.9),text(size:9pt,[#("Bin "+str(row))]))
  for lane in range(8){rect((.8+lane*.5,2.5-row*.9),(1.25+lane*.5,3.05-row*.9),fill:if lane < 6 { pale } else { luma(235) })}
 }
 content((2.8,-.2),text(size:9pt)[32 slots/bin; $B_d=floor((x_d/Delta x-1.5)/4)$.])
 line((5.1,1.7),(6.1,1.7),mark:(end:">"))
 for (row,label,col) in ((0,"x[0..31]",pale),(1,"v[0..31]",warm),(2,"C[0..31]",lilac),(3,"m / V0 / id / substance / phase / flags",pale)){
  rect((6.5,3-row*.8),(12.3,3.55-row*.8),fill:col)
  content((9.4,3.275-row*.8),text(size:9pt,label))
 }
 content((9.4,-.2),text(size:9pt)[Field-major, 128 B aligned; tile $k_d in [4B_d,4B_d+8)$.])
})

#let transfer-figure()=cetz.canvas(length:1cm,{
 import cetz.draw: *
 set-style(stroke:.65pt,mark:(fill:ink))
 content((3,4.3),text(size:10pt,weight:"bold")[P2G · compatible, normalised stencil])
 content((10,4.3),text(size:10pt,weight:"bold")[G2P · full stencil with ghosts])
 for ox in (0,7){
  line((ox+3.9,1.9),(ox+3.9,3.8),stroke:1.5pt+rgb("966747"))
  circle((ox+2.7,2.6),radius:.12,fill:rgb("a66a38"))
  content((ox+2.7,2.2),text(size:9pt)[$x_p$])
  for (x,y) in ((1.8,2.0),(1.8,2.8),(1.8,3.6),(3,2.0),(3,2.8),(3,3.6),(4.2,2.0),(4.2,2.8),(4.2,3.6)){
   let ghost=x>3.9
   line((ox+x - .13,y),(ox+x+.13,y),stroke:(paint:if ghost {rgb("966747")} else {ink},dash:if ghost {"dashed"} else {"solid"}),mark:(end:">"))
   if ox==7 or not ghost {line((ox+2.7,2.6),(ox+x,y),stroke:(paint:luma(180),dash:"dashed"))}
  }
 }
 rect((.2,-.7),(5.9,1.4),fill:pale,radius:.1,stroke:none)
 rect((7.2,-.7),(12.9,1.4),fill:lilac,radius:.1,stroke:none)
 content((3.05,1.0),text(size:9.5pt)[$N'=N/S$; $r'=r-mu$; $D'=sum N'r'r'^T$])
 content((3.05,.35),text(size:9.5pt)[$q_(i a)+=N'm_p(v_(p a)+c_(p a) dot r')$])
 content((3.05,-.3),text(size:9.5pt)[$f_(i a)+=-V_(0p)(tau_p nabla N'_(i a))_a$])
 content((10.05,1.0),text(size:9.5pt)[$v_"pic"=sum N hat(u)^"new"$; $D^"full"=sum N r r^T$])
 content((10.05,.35),text(size:9.5pt)[$delta=v_p-sum N hat(u)^"old"$; $L=sum hat(u)^"new" nabla N^T$])
 content((10.05,-.3),text(size:9.5pt)[Blade-normal ghost component = blade velocity])
 content((6.55,-1.35),text(size:9.5pt)[The cut-band work defect and blade impulse are counted; outside cuts the true-gradient work identity holds.])
})


#let neural-halo-figure()=cetz.canvas(length:1cm,{
 import cetz.draw: *
 set-style(stroke:.65pt,mark:(fill:ink))
 content((6.5,5.0),text(size:10pt,weight:"bold")[Three resolutions; one source-derived dependency chain])
 content((6.5,4.4),text(size:10pt)[$h_"sim"$: simulation; $h_N$: inference lattice; $h_phi=h_N/2$: output and density grid])
 let node(x,label,col,name)={rect((x,2.2),(x+2.2,3.5),fill:col,radius:.08,name:name);content(name+".center",block(width:2cm,align(center,text(size:9pt,label))))}
 node(0,[Output query\ $q in Q$],pale,"query")
 node(2.7,[Feature\ sample],pale,"feature")
 node(5.4,[Contributing\ particle],warm,"particle")
 node(8.1,[Density\ interpolation node],lilac,"density")
 node(10.8,[Normalisation\ source particle],warm,"source")
 for (a,b,label) in (("query","feature",[$3.5h_N$]),("feature","particle",[$2.5h_N$]),("particle","density",[$0.5h_N$]),("density","source",[$2.5h_N$])){
  line(a+".east",b+".west",mark:(end:">"))
  content((a+".center",50%,b+".center"),anchor:"south",padding:.9,text(size:9.5pt,label))
 }
 content((6.5,1.6),text(size:10pt)[$R_("dep",infinity)<= (3.5+2.5+0.5+2.5)h_N=9h_N$ · shipped stride-2 configuration])
 rect((0,-.8),(6.1,.9),fill:pale,radius:.08)
 content((3.05,.05),block(width:5.8cm,align(center,text(size:9.5pt)[Full cache: reference inputs\ Band cache: measure lost inputs\ Source coverage for every $q in Q$.])))
 rect((6.9,-.8),(13,.9),fill:lilac,radius:.08)
 content((9.95,.05),block(width:5.8cm,align(center,text(size:9.5pt)[$Q$: queries plus extraction halo\ $9h_N$: full $L^infinity$ dependency,\ including density normalisation.])))
 content((6.5,-1.4),text(size:9.5pt)[Negative interior output fill cannot restore lost CNN inputs or density denominators.])
 content((6.5,-2.0),text(size:9.5pt)[A surface-distance particle band needs its own coverage check; this bound does not set $h_"sim"$ or widen that band.])
})

#let validation-figure()=cetz.canvas(length:1cm,{
 import cetz.draw: *
 set-style(stroke:.65pt,mark:(fill:ink))
 let node(x,y,w,h,label,col,name)={rect((x,y),(x+w,y+h),fill:col,radius:.08,name:name);content(name+".center",block(width:(w - .3)*1cm,align(center,text(size:9.5pt,label))))}
 node(0,4.6,6,2.0,[Independent Python reference\ 2D MAC: direct matrices and solves\ separate 3D algebraic fixtures\ independent assembly],pale,"python")
 node(7,4.6,6,2.0,[Ten Lean contracts: exact models\ F1–F4: transfers and resampling\ F5–F7: pressure and viscosity\ F8–F10: sums, rollback, tile bounds],lilac,"lean")
 node(3.5,1.7,6,1.5,[C++ CPU implementation\ compare independent fixtures\ check hypotheses and tolerances],warm,"cpu")
 node(3.5,-.9,6,1.5,[CUDA on real devices\ parity, guards, failure injection\ quality / latency / memory],warm,"cuda")
 line("python.south",(3,3.8),(5,3.8),"cpu.north",mark:(end:">"))
 line("lean.south",(10,3.8),(8,3.8),"cpu.north",mark:(end:">"))
 content((1.65,3.5),text(size:9pt)[Numerical oracles])
 content((11.25,3.5),text(size:9pt)[Named assumptions])
 line("cpu.south","cuda.north",mark:(end:">"))
 content((6.5,1.1),text(size:9pt)[Implementation bridge and runtime checks])
 content((6.5,-1.65),text(size:9.5pt)[Four separate results: compiled contracts · independent fixtures · implementation checks · device measurements.])
 content((6.5,-2.2),text(size:9.5pt)[Exact proofs do not certify IEEE arithmetic, C++ memory safety, CUDA races, geometry or visual quality.])
})

#let meshing-dag-figure()=cetz.canvas(length:1cm,{
 import cetz.draw: *
 set-style(stroke:.65pt,mark:(fill:ink))
 let node(x,y,label,col,name)={rect((x,y),(x+3.6,y+1.6),fill:col,radius:.08,name:name);content(name+".center",block(width:3.3cm,align(center,text(size:9pt,label))))}
 node(0,3.4,[Simulate · GPU\ particles, velocity\ `sdf_sim`, source events],pale,"sim")
 node(4.7,3.4,[Mesh input field\ classical baseline\ read-only sim cache],lilac,"field")
 node(9.4,3.4,[Mostly CPU tracker\ advect + project\ repair / extract / export],warm,"tracker")
 node(4.7,6.0,[Optional neural field\ GPU features + inference\ full-vs-band comparison],lilac,"neural")
 node(9.4,6.0,[Previous Mesh state\ accepted tracker state\ temporal correspondence],warm,"previous")
 node(0,.6,[Whitewater\ consume substep events\ foam / bubbles / spray],pale,"ww")
 node(4.7,.6,[Volumes\ simulation + whitewater\ density and SDF export],pale,"vol")
 node(9.4,.6,[Multires / final export\ consume tracked Mesh\ render assets],warm,"multires")
 line("sim.east","field.west",mark:(end:">"))
 line("field.east","tracker.west",mark:(end:">"))
 line("field.north","neural.south",mark:(end:">"),stroke:(dash:"dashed"))
 line("neural.east",(8.8,6.7),(8.8,5.2),(11.2,5.2),"tracker.north",mark:(end:">"),stroke:(dash:"dashed"))
 line("previous.south","tracker.north",mark:(end:">"))
 line("tracker.south","multires.north",mark:(end:">"))
 line("sim.south","ww.north",mark:(end:">"))
 line("sim.south",(3.2,2.7),(6.5,2.7),"vol.north",mark:(end:">"))
 line("ww.east","vol.west",mark:(end:">"))
 content((6.5,-.25),text(size:9.5pt)[Simulate never reads Mesh. Whitewater and Volumes do not require the tracked mesh.])
 content((6.5,-.85),text(size:9.5pt)[Optional neural GPU work shares device memory and compute with Simulate: schedule and measure contention.])
})
