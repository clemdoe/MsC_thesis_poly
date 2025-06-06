*DECK SNFT12
      SUBROUTINE SNFT12(NUN,NGEFF,IMPX,INCONV,NGIND,LX,LY,IELEM,NMAT,
     1 NPQ,NSCT,MAT,VOL,TOTAL,NCODE,ZCODE,QEXT,LFIXUP,DU,DE,W,MRM,MRMY,
     2 DB,DA,FUNKNO,ISBS,NBS,ISBSM,BS,MAXL,MN,DN)
*
*-----------------------------------------------------------------------
*
*Purpose:
* Perform one inner iteration for solving SN equations in 2D Cartesian
* geometry for the HODD method. Energy-angle multithreading. Albedo
* boundary conditions.
*
*Copyright:
* Copyright (C) 2020 Ecole Polytechnique de Montreal
* This library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 2.1 of the License, or (at your option) any later version
*
*Author(s): A. Hebert
*
*Parameters: input
* NUN     total number of unknowns in vector FUNKNO.
* NGEFF   number of energy groups processed in parallel.
* IMPX    print flag (equal to zero for no print).
* INCONV  energy group convergence flag (set to .FALSE. if converged).
* NGIND   energy group indices assign to the NGEFF set.
* LX      number of meshes along X axis.
* LY      number of meshes along Y axis.
* IELEM   measure of order of the spatial approximation polynomial:
*         =1 constant - classical diamond scheme - default for HODD;
*         =2 linear;
*         =3 parabolic.
* NMAT    number of material mixtures.
* NPQ     number of SN directions in four octants (including zero-weight
*         directions).
* NSCT    maximum number of spherical harmonics moments of the flux.
* MAT     material mixture index in each region.
* VOL     volumes of each region.
* TOTAL   macroscopic total cross sections.
* NCODE   boundary condition indices.
* ZCODE   albedos.
* QEXT    Legendre components of the fixed source.
* LFIXUP  flag to enable negative flux fixup.
* DU      first direction cosines ($\\mu$).
* DE      second direction cosines ($\\eta$).
* W       weights.
* MRM     quadrature index.
* MRMY    quadrature index.
* DB      diamond-scheme parameter.
* DA      diamond-scheme parameter.
* MN      moment-to-discrete matrix.
* DN      discrete-to-moment matrix.
*
*Parameters: input/output
* FUNKNO  Legendre components of the flux and boundary fluxes.
*
*-----------------------------------------------------------------------
*
#if defined(_OPENMP)
      USE omp_lib
#endif
*----
*  SUBROUTINE ARGUMENTS
*----
      INTEGER NUN,NGEFF,IMPX,NGIND(NGEFF),LX,LY,IELEM,NMAT,NPQ,NSCT,
     1 MAT(LX,LY),NCODE(4),MRM(NPQ),MRMY(NPQ),ISBS,NBS,
     2 ISBSM(4*ISBS,NPQ*ISBS,NGEFF*ISBS),MAXL
      LOGICAL INCONV(NGEFF)
      REAL VOL(LX,LY),TOTAL(0:NMAT,NGEFF),ZCODE(4),QEXT(NUN,NGEFF),
     1 DU(NPQ),DE(NPQ),W(NPQ),DB(LX,NPQ),DA(LX,LY,NPQ),
     2 FUNKNO(NUN,NGEFF),BS(MAXL*ISBS,NBS*ISBS),MN(NPQ,NSCT),
     3 DN(NSCT,NPQ)
      LOGICAL LFIXUP
*----
*  LOCAL VARIABLES
*----
      INTEGER NPQD(4),IIND(4),P
      DOUBLE PRECISION Q(IELEM**2),Q2(IELEM**2,(IELEM**2)+1),XNJ(IELEM),
     1 VT,CONST0,CONST1,CONST2
      PARAMETER(IUNOUT=6,RLOG=1.0E-8,PI=3.141592654)
*----
*  ALLOCATABLE ARRAYS
*----
      INTEGER, ALLOCATABLE, DIMENSION(:,:) :: INDANG
      DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:,:,:) :: FLUX
      DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:,:,:,:) :: FLUX_G
      DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:) :: XNI
*----
*  SCRATCH STORAGE ALLOCATION
*----
      ALLOCATE(INDANG(NPQ,4))
      ALLOCATE(XNI(IELEM,LY),FLUX(IELEM**2,NSCT,LX,LY))
      ALLOCATE(FLUX_G(IELEM**2,NSCT,LX,LY,NGEFF))
*----
*  DEFINITION OF CONSTANTS.
*----
      L4=IELEM*IELEM*LX*LY*NSCT
      CONST0=2.0D0*DSQRT(3.0D0)
      CONST1=2.0D0*DSQRT(5.0D0)
      CONST2=2.0D0*DSQRT(15.0D0)
*----
*  PARAMETER VALIDATION.
*----
      IF(IELEM.GT.4) CALL XABORT('SNFT12: INVALID IELEM (DIAM) VALUE. '
     1 //'CHECK INPUT DATA FILE.')
      FLUX_G(:IELEM**2,:NSCT,:LX,:LY,:NGEFF)=0.0D0
*----
*  SET OCTANT SWAPPING ORDER.
*----
      NPQD(:4)=0
      INDANG(:NPQ,:4)=0
      IIND(:)=0
      DO M=1,NPQ
        VU=DU(M)
        VE=DE(M)
        IF(W(M).EQ.0) CYCLE
        IF((VU.GE.0.0).AND.(VE.GE.0.0)) THEN
          IND=1
          JND=4
        ELSE IF((VU.LE.0.0).AND.(VE.GE.0.0)) THEN
          IND=2
          JND=3
        ELSE IF((VU.LE.0.0).AND.(VE.LE.0.0)) THEN
          IND=3
          JND=1
        ELSE
          IND=4
          JND=2
        ENDIF
        IIND(JND)=IND
        NPQD(IND)=NPQD(IND)+1
        INDANG(NPQD(IND),IND)=M
      ENDDO
*----
*  MAIN LOOP OVER OCTANTS.
*----
      DO 190 JND=1,4
      IND=IIND(JND)
*----
*  PRELIMINARY LOOPS FOR SETTING BOUNDARY CONDITIONS.
*----
*$OMP  PARALLEL DO
*$OMP1 PRIVATE(M,IG,WEIGHT,VU,VE,M1,E1,IOF,JOF,IEL,I,J)
*$OMP2 SHARED(FUNKNO) COLLAPSE(2)
      DO 70 IG=1,NGEFF
      DO 60 IPQD=1,NPQD(IND)
      IF(.NOT.INCONV(IG)) GO TO 60
      M=INDANG(IPQD,IND)
      WEIGHT=W(M)
      VU=DU(M)
      VE=DE(M)
      IF(VU.GT.0.0)THEN
         M1=MRM(M)
         IF((NCODE(1).NE.4))THEN
            DO IEL=1,IELEM
               DO J=1,LY
                  IOF=((M-1)*LY+(J-1))*IELEM+IEL
                  JOF=((M1-1)*LY+(J-1))*IELEM+IEL
                  FUNKNO(L4+IOF,IG)=FUNKNO(L4+JOF,IG)
               ENDDO
            ENDDO
         ENDIF
      ELSEIF(VU.LT.0.0)THEN
         M1=MRM(M)
         IF((NCODE(2).NE.4))THEN
            DO IEL=1,IELEM
               DO J=1,LY
                  IOF=((M-1)*LY+(J-1))*IELEM+IEL
                  JOF=((M1-1)*LY+(J-1))*IELEM+IEL
                  FUNKNO(L4+IOF,IG)=FUNKNO(L4+JOF,IG)
               ENDDO
            ENDDO
         ENDIF
      ENDIF
      IF(VE.GT.0.0)THEN
         M1=MRMY(M)
         IF((NCODE(3).NE.4))THEN
            DO IEL=1,IELEM
               DO I=1,LX
                  IOF=((M-1)*LX+(I-1))*IELEM+IEL
                  JOF=((M1-1)*LX+(I-1))*IELEM+IEL
                  FUNKNO(L4+IELEM*LY*NPQ+IOF,IG)=
     >               FUNKNO(L4+IELEM*LY*NPQ+JOF,IG)
               ENDDO
            ENDDO
         ENDIF
      ELSEIF(VE.LT.0.0)THEN
         M1=MRMY(M)
         IF((NCODE(4).NE.4))THEN
            DO IEL=1,IELEM
               DO I=1,LX
                  IOF=((M-1)*LX+(I-1))*IELEM+IEL
                  JOF=((M1-1)*LX+(I-1))*IELEM+IEL
                  FUNKNO(L4+IELEM*LY*NPQ+IOF,IG)=
     >               FUNKNO(L4+IELEM*LY*NPQ+JOF,IG)
               ENDDO
            ENDDO
         ENDIF
      ENDIF
   60 CONTINUE
   70 CONTINUE
*$OMP END PARALLEL DO
*----
*  MAIN SWAPPING LOOPS FOR SN FLUX CALCULATION
*----
*$OMP  PARALLEL DO
*$OMP1 PRIVATE(ITID,FLUX,M,IG,XNI,XNJ,Q,Q2,IOF,IER,II,JJ,IEL,JEL,I,J,K)
*$OMP2 PRIVATE(VT) SHARED(FUNKNO) REDUCTION(+:FLUX_G)
*$OMP3 COLLAPSE(2)
      DO 180 IG=1,NGEFF
      DO 170 IPQD=1,NPQD(IND)
#if defined(_OPENMP)
        ITID=omp_get_thread_num()
#else
        ITID=0
#endif
      IF(IMPX.GT.5) WRITE(IUNOUT,400) ITID,NGIND(IG),IPQD
      IF(.NOT.INCONV(IG)) GO TO 170
      M=INDANG(IPQD,IND)
      FLUX(:IELEM**2,:NSCT,:LX,:LY)=0.0D0
      IF(W(M).EQ.0.0) GO TO 170
*----
*  LOOP OVER X- AND Y-DIRECTED AXES.
*----
      DO 155 II=1,LX
      I=II
      IF((IND.EQ.2).OR.(IND.EQ.3)) I=LX+1-I
      DO 100 IEL=1,IELEM
      IOF=(M-1)*IELEM*LX+(I-1)*IELEM+IEL
      IF((IND.EQ.1).OR.(IND.EQ.2)) THEN
         XNJ(IEL)=FUNKNO(L4+IELEM*LY*NPQ+IOF,IG)*ZCODE(3)
      ELSE
         XNJ(IEL)=FUNKNO(L4+IELEM*LY*NPQ+IOF,IG)*ZCODE(4)
      ENDIF
  100 CONTINUE
      IF(ISBS.EQ.1) THEN
        IF((IND.EQ.3.OR.IND.EQ.4).AND.ISBSM(4,M,IG).NE.0) THEN
          XNJ(1)=XNJ(1)+BS(I,ISBSM(4,M,IG))
        ELSE IF((IND.EQ.1.OR.IND.EQ.2).AND.ISBSM(3,M,IG).NE.0) THEN
          XNJ(1)=XNJ(1)+BS(I,ISBSM(3,M,IG))
        ENDIF
      ENDIF
      DO 140 JJ=1,LY
      J=JJ
      IF((IND.EQ.3).OR.(IND.EQ.4)) J=LY+1-J
      DO 105 IEL=1,IELEM
      IF(II.EQ.1) THEN
         IOF=(M-1)*IELEM*LY+(J-1)*IELEM+IEL
         IF((IND.EQ.1).OR.(IND.EQ.4)) THEN
            XNI(IEL,J)=FUNKNO(L4+IOF,IG)*ZCODE(1)
         ELSE
            XNI(IEL,J)=FUNKNO(L4+IOF,IG)*ZCODE(2)
         ENDIF
      ENDIF
  105 CONTINUE
      IF(ISBS.EQ.1.AND.II.EQ.1) THEN
        IF((IND.EQ.2.OR.IND.EQ.3).AND.ISBSM(2,M,IG).NE.0) THEN
          XNI(1,J)=XNI(1,J)+BS(J,ISBSM(2,M,IG))
        ELSE IF((IND.EQ.1.OR.IND.EQ.4).AND.ISBSM(1,M,IG).NE.0) THEN
          XNI(1,J)=XNI(1,J)+BS(J,ISBSM(1,M,IG))
        ENDIF
      ENDIF 
      IF(MAT(I,J).EQ.0) GO TO 140
*------
      DO 115 IEL=1,IELEM**2
      Q(IEL)=0.0D0
      DO 110 P=1,NSCT
      IOF=((J-1)*LX*NSCT+(I-1)*NSCT+(P-1))*IELEM*IELEM+IEL
      Q(IEL)=Q(IEL)+QEXT(IOF,IG)*MN(M,P)
  110 CONTINUE
  115 CONTINUE
      VT=VOL(I,J)*TOTAL(MAT(I,J),IG)
      Q2(:IELEM**2,:(IELEM**2)+1)=0.0D0
      IF(IELEM.EQ.1) THEN
         Q2(1,1)=2.0D0*ABS(DA(I,J,M))+2.0D0*ABS(DB(I,M))+VT
*        ------
         Q2(1,2)=2.0D0*ABS(DA(I,J,M))*XNI(1,J)+2.0D0*ABS(DB(I,M))
     1           *XNJ(1)+VOL(I,J)*Q(1)
      ELSE IF(IELEM.EQ.2) THEN
         Q2(1,1)=VT
         Q2(2,1)=CONST0*DA(I,J,M)
         Q2(2,2)=-VT-6.0D0*ABS(DA(I,J,M))
         Q2(3,1)=CONST0*DB(I,M)
         Q2(3,3)=-VT-6.0D0*ABS(DB(I,M))
         Q2(4,2)=-CONST0*DB(I,M)
         Q2(4,3)=-CONST0*DA(I,J,M)
         Q2(4,4)=VT+6.0D0*ABS(DA(I,J,M))+6.0D0*ABS(DB(I,M))
*        ------
         Q2(1,5)=VOL(I,J)*Q(1)
         Q2(2,5)=-VOL(I,J)*Q(2)+CONST0*DA(I,J,M)*XNI(1,J)
         Q2(3,5)=-VOL(I,J)*Q(3)+CONST0*DB(I,M)*XNJ(1)
         Q2(4,5)=VOL(I,J)*Q(4)-CONST0*DA(I,J,M)*XNI(2,J)-CONST0*
     1           DB(I,M)*XNJ(2)
      ELSE IF(IELEM.EQ.3) THEN
         Q2(1,1)=VT+2.0D0*ABS(DA(I,J,M))+2.0D0*ABS(DB(I,M))
         Q2(2,2)=-VT-2.0D0*ABS(DB(I,M))
         Q2(3,1)=CONST1*ABS(DA(I,J,M))
         Q2(3,2)=-CONST2*DA(I,J,M)
         Q2(3,3)=VT+1.0D1*ABS(DA(I,J,M))+2.0D0*ABS(DB(I,M))
         Q2(4,4)=-VT-2.0D0*ABS(DA(I,J,M))
         Q2(5,5)=VT
         Q2(6,4)=-CONST1*ABS(DA(I,J,M))
         Q2(6,5)=CONST2*DA(I,J,M)
         Q2(6,6)=-VT-1.0D1*ABS(DA(I,J,M))
         Q2(7,1)=CONST1*ABS(DB(I,M))
         Q2(7,4)=-CONST2*DB(I,M)
         Q2(7,7)=VT+2.0D0*ABS(DA(I,J,M))+1.0D1*ABS(DB(I,M))
         Q2(8,2)=-CONST1*ABS(DB(I,M))
         Q2(8,5)=CONST2*DB(I,M)
         Q2(8,8)=-VT-1.0D1*ABS(DB(I,M))
         Q2(9,3)=CONST1*ABS(DB(I,M))
         Q2(9,6)=-CONST2*DB(I,M)
         Q2(9,7)=CONST1*ABS(DA(I,J,M))
         Q2(9,8)=-CONST2*DA(I,J,M)
         Q2(9,9)=VT+1.0D1*ABS(DA(I,J,M))+1.0D1*ABS(DB(I,M))
*        ------
         Q2(1,10)=VOL(I,J)*Q(1)+2.0D0*ABS(DA(I,J,M))*XNI(1,J)+2.0D0*
     1            ABS(DB(I,M))*XNJ(1)
         Q2(2,10)=-VOL(I,J)*Q(2)-2.0D0*ABS(DB(I,M))*XNJ(2)
         Q2(3,10)=VOL(I,J)*Q(3)+CONST1*ABS(DA(I,J,M))*XNI(1,J)+2.0D0*
     1            ABS(DB(I,M))*XNJ(3)
         Q2(4,10)=-VOL(I,J)*Q(4)-2.0D0*ABS(DA(I,J,M))*XNI(2,J)
         Q2(5,10)=VOL(I,J)*Q(5)
         Q2(6,10)=-VOL(I,J)*Q(6)-CONST1*ABS(DA(I,J,M))*XNI(2,J)
         Q2(7,10)=VOL(I,J)*Q(7)+2.0D0*ABS(DA(I,J,M))*XNI(3,J)+CONST1*
     1            ABS(DB(I,M))*XNJ(1)
         Q2(8,10)=-VOL(I,J)*Q(8)-CONST1*ABS(DB(I,M))*XNJ(2)
         Q2(9,10)=VOL(I,J)*Q(9)+CONST1*ABS(DA(I,J,M))*XNI(3,J)+CONST1*
     1            ABS(DB(I,M))*XNJ(3)
      ELSE IF(IELEM.EQ.4) THEN
         Q2(1,1) = VT
         Q2(2,1) = 2*3**(0.5D0)*DA(I,J,M)
         Q2(2,2) = - VT - 6*ABS(DA(I,J,M))
         Q2(3,3) = VT
         Q2(4,1) = 2*7**(0.5D0)*DA(I,J,M)
         Q2(4,2) = -2*21**(0.5D0)*ABS(DA(I,J,M))
         Q2(4,3) = 2*35**(0.5D0)*DA(I,J,M)
         Q2(4,4) = - VT - 14*ABS(DA(I,J,M))
         Q2(5,1) = 2*3**(0.5D0)*DB(I,M)
         Q2(5,5) = - VT - 6*ABS(DB(I,M))
         Q2(6,2) = -2*3**(0.5D0)*DB(I,M)
         Q2(6,5) = -2*3**(0.5D0)*DA(I,J,M)
         Q2(6,6) = VT + 6*ABS(DB(I,M)) + 6*ABS(DA(I,J,M))
         Q2(7,3) = 2*3**(0.5D0)*DB(I,M)
         Q2(7,7) = - VT - 6*ABS(DB(I,M))
         Q2(8,4) = -2*3**(0.5D0)*DB(I,M)
         Q2(8,5) = -2*7**(0.5D0)*DA(I,J,M)
         Q2(8,6) = 2*21**(0.5D0)*ABS(DA(I,J,M))
         Q2(8,7) = -2*35**(0.5D0)*DA(I,J,M)
         Q2(8,8) = VT + 6*ABS(DB(I,M)) + 14*ABS(DA(I,J,M))
         Q2(9,9) = VT
         Q2(10,9) = 2*3**(0.5D0)*DA(I,J,M)
         Q2(10,10) = - VT - 6*ABS(DA(I,J,M))
         Q2(11,11) = VT
         Q2(12,9) = 2*7**(0.5D0)*DA(I,J,M)
         Q2(12,10) = -2*21**(0.5D0)*ABS(DA(I,J,M))
         Q2(12,11) = 2*35**(0.5D0)*DA(I,J,M)
         Q2(12,12) = - VT - 14*ABS(DA(I,J,M))
         Q2(13,1) = 2*7**(0.5D0)*DB(I,M)
         Q2(13,5) = -2*21**(0.5D0)*ABS(DB(I,M))
         Q2(13,9) = 2*35**(0.5D0)*DB(I,M)
         Q2(13,13) = - VT - 14*ABS(DB(I,M))
         Q2(14,2) = -2*7**(0.5D0)*DB(I,M)
         Q2(14,6) = 2*21**(0.5D0)*ABS(DB(I,M))
         Q2(14,10) = -2*35**(0.5D0)*DB(I,M)
         Q2(14,13) = -2*3**(0.5D0)*DA(I,J,M)
         Q2(14,14) = VT + 14*ABS(DB(I,M)) + 6*ABS(DA(I,J,M))
         Q2(15,3) = 2*7**(0.5D0)*DB(I,M)
         Q2(15,7) = -2*21**(0.5D0)*ABS(DB(I,M))
         Q2(15,11) = 2*35**(0.5D0)*DB(I,M)
         Q2(15,15) = - VT - 14*ABS(DB(I,M))
         Q2(15,16) = -2*35**(0.5D0)*DA(I,J,M)
         Q2(16,4) = -2*7**(0.5D0)*DB(I,M)
         Q2(16,8) = 2*21**(0.5D0)*ABS(DB(I,M))
         Q2(16,12) = -2*35**(0.5D0)*DB(I,M)
         Q2(16,13) = -2*7**(0.5D0)*DA(I,J,M)
         Q2(16,14) = 2*21**(0.5D0)*ABS(DA(I,J,M))
         Q2(16,15) = -2*35**(0.5D0)*DA(I,J,M)
         Q2(16,16) = VT + 14*ABS(DB(I,M)) + 14*ABS(DA(I,J,M))
*        ------
         Q2(1,17) = Q(1)*VOL(I,J)
         Q2(2,17) = -Q(2)*VOL(I,J)
         Q2(3,17) = Q(3)*VOL(I,J)
         Q2(4,17) = -Q(4)*VOL(I,J)
         Q2(5,17) = -Q(5)*VOL(I,J)
         Q2(6,17) = Q(6)*VOL(I,J)
         Q2(7,17) = -Q(7)*VOL(I,J)
         Q2(8,17) = Q(8)*VOL(I,J)
         Q2(9,17) = Q(9)*VOL(I,J)
         Q2(10,17) = -Q(10)*VOL(I,J)
         Q2(11,17) = Q(11)*VOL(I,J)
         Q2(12,17) = -Q(12)*VOL(I,J)
         Q2(13,17) = -Q(13)*VOL(I,J)
         Q2(14,17) = Q(14)*VOL(I,J)
         Q2(15,17) = -Q(15)*VOL(I,J)
         Q2(16,17) = Q(16)*VOL(I,J)

         Q2(2,17) = Q2(2,17) + 2*3**(0.5D0)*DA(I,J,M)*XNI(1,J)
         Q2(4,17) = Q2(4,17) + 2*7**(0.5D0)*DA(I,J,M)*XNI(1,J)
         Q2(5,17) = Q2(5,17) + 2*3**(0.5D0)*DB(I,M)*XNJ(1)
         Q2(6,17) = Q2(6,17) + (- 2*3**(0.5D0)*DB(I,M)*XNJ(2) -
     >   2*3**(0.5D0)*DA(I,J,M)*XNI(2,J))
         Q2(7,17) = Q2(7,17) + 2*3**(0.5D0)*DB(I,M)*XNJ(3)
         Q2(8,17) = Q2(8,17) + (- 2*3**(0.5D0)*DB(I,M)*XNJ(4) -
     >   2*7**(0.5D0)*DA(I,J,M)*XNI(2,J))
         Q2(10,17) = Q2(10,17) + 2*3**(0.5D0)*DA(I,J,M)*XNI(3,J)
         Q2(12,17) = Q2(12,17) + 2*7**(0.5D0)*DA(I,J,M)*XNI(3,J)
         Q2(13,17) = Q2(13,17) + 2*7**(0.5D0)*DB(I,M)*XNJ(1)
         Q2(14,17) = Q2(14,17) + (- 2*7**(0.5D0)*DB(I,M)*XNJ(2) -
     >   2*3**(0.5D0)*DA(I,J,M)*XNI(4,J))
         Q2(15,17) = Q2(15,17) + 2*7**(0.5D0)*DB(I,M)*XNJ(3)
         Q2(16,17) = Q2(16,17) + (- 2*7**(0.5D0)*DB(I,M)*XNJ(4) -
     >   2*7**(0.5D0)*DA(I,J,M)*XNI(4,J))
      ENDIF
*
      DO 125 IEL=1,IELEM**2
      DO 120 JEL=IEL+1,IELEM**2
      Q2(IEL,JEL)=Q2(JEL,IEL)
  120 CONTINUE
  125 CONTINUE
*
      CALL ALSBD(IELEM**2,1,Q2,IER,IELEM**2)
      IF(IER.NE.0) CALL XABORT('SNFT12: SINGULAR MATRIX.')
*
      IF(IELEM.EQ.1) THEN
         IF(LFIXUP.AND.(Q2(1,2).LE.RLOG)) Q2(1,2)=0.0
         XNI(1,J)=2.0D0*Q2(1,2)-XNI(1,J)
         XNJ(1)=2.0D0*Q2(1,2)-XNJ(1)
         IF(LFIXUP.AND.(XNI(1,J).LE.RLOG)) XNI(1,J)=0.0
         IF(LFIXUP.AND.(XNJ(1).LE.RLOG)) XNJ(1)=0.0
      ELSE IF(IELEM.EQ.2) THEN
         XNI(1,J)=XNI(1,J)+SIGN(1.0,DU(M))*CONST0*Q2(2,5)
         XNI(2,J)=XNI(2,J)+SIGN(1.0,DU(M))*CONST0*Q2(4,5)
         XNJ(1)=XNJ(1)+SIGN(1.0,DE(M))*CONST0*Q2(3,5)
         XNJ(2)=XNJ(2)+SIGN(1.0,DE(M))*CONST0*Q2(4,5)
      ELSE IF(IELEM.EQ.3) THEN
         XNI(1,J)=2.0D0*Q2(1,10)+CONST1*Q2(3,10)-XNI(1,J)
         XNI(2,J)=2.0D0*Q2(4,10)+CONST1*Q2(6,10)-XNI(2,J)
         XNI(3,J)=2.0D0*Q2(7,10)+CONST1*Q2(9,10)-XNI(3,J)
         XNJ(1)=2.0D0*Q2(1,10)+CONST1*Q2(7,10)-XNJ(1)
         XNJ(2)=2.0D0*Q2(2,10)+CONST1*Q2(8,10)-XNJ(2)
         XNJ(3)=2.0D0*Q2(3,10)+CONST1*Q2(9,10)-XNJ(3)
      ELSE IF(IELEM.EQ.4) THEN
         XNI(1,J) = XNI(1,J) + SIGN(1.0,DU(M))*2*3
     >   **(0.5D0)*Q2(02,17) + SIGN(1.0,DU(M))*2*7
     >   **(0.5D0)*Q2(04,17)
         XNI(2,J) = XNI(2,J) + SIGN(1.0,DU(M))*2*3
     >   **(0.5D0)*Q2(06,17) + SIGN(1.0,DU(M))*2*7
     >   **(0.5D0)*Q2(08,17)
         XNI(3,J) = XNI(3,J) + SIGN(1.0,DU(M))*2*3
     >   **(0.5D0)*Q2(10,17) + SIGN(1.0,DU(M))*2*7
     >   **(0.5D0)*Q2(12,17)
         XNI(4,J) = XNI(4,J) + SIGN(1.0,DU(M))*2*3
     >   **(0.5D0)*Q2(14,17) + SIGN(1.0,DU(M))*2*7
     >   **(0.5D0)*Q2(16,17)
         XNJ(1)   =   XNJ(1) + SIGN(1.0,DE(M))*2*7
     >   **(0.5D0)*Q2(13,17) + SIGN(1.0,DE(M))*2*3
     >   **(0.5D0)*Q2(05,17)
         XNJ(2)   =   XNJ(2) + SIGN(1.0,DE(M))*2*7
     >   **(0.5D0)*Q2(14,17) + SIGN(1.0,DE(M))*2*3
     >   **(0.5D0)*Q2(06,17)
         XNJ(3)   =   XNJ(3) + SIGN(1.0,DE(M))*2*7
     >   **(0.5D0)*Q2(15,17) + SIGN(1.0,DE(M))*2*3
     >   **(0.5D0)*Q2(07,17)
         XNJ(4)   =   XNJ(4) + SIGN(1.0,DE(M))*2*7
     >   **(0.5D0)*Q2(16,17) + SIGN(1.0,DE(M))*2*3
     >   **(0.5D0)*Q2(08,17)
      ENDIF
*
      DO 135 P=1,NSCT
      DO 130 IEL=1,IELEM**2
      FLUX(IEL,P,I,J)=FLUX(IEL,P,I,J)+Q2(IEL,IELEM**2+1)*DN(P,M)
  130 CONTINUE
  135 CONTINUE
*------
  140 CONTINUE
      DO 150 IEL=1,IELEM
      IOF=(M-1)*IELEM*LX+(I-1)*IELEM+IEL
      FUNKNO(L4+IELEM*LY*NPQ+IOF,IG)=REAL(XNJ(IEL))
  150 CONTINUE
*--
  155 CONTINUE
      DO 165 J=1,LY
      DO 160 IEL=1,IELEM
      IOF=(M-1)*IELEM*LY+(J-1)*IELEM+IEL
      FUNKNO(L4+IOF,IG)=REAL(XNI(IEL,J))
  160 CONTINUE
  165 CONTINUE
      FLUX_G(:,:,:,:,IG)=FLUX_G(:,:,:,:,IG)+FLUX(:,:,:,:)
  170 CONTINUE
  180 CONTINUE
*$OMP END PARALLEL DO
  190 CONTINUE
      DO 200 IG=1,NGEFF
        IF(.NOT.INCONV(IG)) GO TO 200
        FUNKNO(:L4,IG)=
     1  RESHAPE(REAL(FLUX_G(:IELEM**2,:NSCT,:LX,:LY,IG)), (/ L4 /) )
  200 CONTINUE
*----
*  SCRATCH STORAGE DEALLOCATION
*----
      DEALLOCATE(XNI,FLUX_G,FLUX,INDANG)
      RETURN
  400 FORMAT(16H SNFT12: thread=,I8,12H --->(group=,I4,7H angle=,I4,1H))
      END
