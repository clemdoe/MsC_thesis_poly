*DECK SPHEQU
      SUBROUTINE SPHEQU(NBMIX2,IPTRK2,IFTRAK,IPMACR,IPFLX,CDOOR,NSPH,
     1 KSPH,MAXIT,MAXNBI,EPSPH,IPRINT,IMC,NGCOND,NMERGE,NALBP,ISCAT,
     2 NREG2,NUN2,MAT2,VOL2,KEY2,MERG2,ILK,CTITRE,IGRMIN,IGRMAX,SPH)
*
*-----------------------------------------------------------------------
*
*Purpose:
* Calculation of the SPH factors for the homogenization of any geometry
* using a transport-transport or transport-diffusion equivalence
* technique. The macro-calculation can be performed using a standard
* solution door of Dragon.
*
*Copyright:
* Copyright (C) 2002 Ecole Polytechnique de Montreal
* This library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 2.1 of the License, or (at your option) any later version
*
*Author(s): A. Hebert
*
*Parameters: input
* NBMIX2  number of macro-mixtures. Equal to MAX(MAT2(IREG)) for
*         (IREG .le. NREG2).
* IPTRK2  pointer to the tracking of the macro-geometry (L_TRACK
*         signature).
* IFTRAK  unit number of the sequential binary tracking file.
* IPMACR  pointer to the reference macrolib (L_MACROLIB signature).
* IPFLX   pointer towards an initialization flux (L_FLUX signature).
* CDOOR   tracking operator used to track the macro-geometry.
* NSPH    type of SPH algorithm:
*         =2 homogeneous macro-calculation (non-iterative procedure
*            or Hebert-Benoist SPH-5 procedure);
*         =3 any type of pij macro-calculation;
*         =4 any type of diffusion, SN, PN or SPN macro-calculation;
* KSPH    type of SPH factor normalization:
*         <0 asymptotic normatization with respect to mixture -KSPH;
*         =1 average flux normalization;
*         =2 Selengut normalization using ALBS information;
*         =3 Selengut normalization using FD_B boundary fluxes;
*         =4 generalized Selengut normalization (EDF type);
*         =5 Selengut normalization with surface leakage;
*         =6 Selengut with water gap normalization;
*         =7 average flux normalization in fissile zones.
*         The Hebert-Benoist procedure is used if NSPH=2 and KSPH=5.
* MAXIT   maximum number of SPH iterations.
* MAXNBI  acceptable number of SPH iterations with an increase in
*         convergence error before aborting.
* EPSPH   convergence criterion for stopping the SPH iterations.
* IPRINT  print flag (equal to 0 for no print).
* IMC     type of macro-calculation (=1 diffusion or SPN; 
*         =2 other options;
*         =3 type PIJ with Bell acceleration).
* NGCOND  number of condensed groups.
* NMERGE  number of merged regions (equal to 1 or to the number of
*         different flux components in the macro-calculation).
* NALBP   number of physical albedos.
* ISCAT   scattering anisotropy in the reference set of cross sections
*         (=1 isotropic in LAB; =2 linearly-anisotropic in LAB).
* NREG2   number of macro-regions (in the macro-calculation).
* NUN2    number of unknowns in a one-group macro-calculation.
* MAT2    mixture index per macro-region.
* VOL2    volume of macro-regions.
* KEY2    pointer to flux values in unknown vector.
* MERG2   index of merged macro-regions per macro-mixture.
* ILK     leakage switch.
* CTITRE  title.
* IGRMIN  first group to process.
* IGRMAX  last group to process.
*
*Parameters: output
* SPH     SPH homogenization factors.
*
*References(s):
*  A. Hebert, 'A Consistent Technique for the Pin-by-Pin Homogenization
*  of a Pressurized Water Reactor Assembly', Nucl. Sci. Eng., 113, 227
*  (1993).
*
*  A. Hebert and P. Benoist, 'A Consistent Technique for the Global
*  Homogenization of a Pressurized Water Reactor Assembly', Nucl. Sci.
*  Eng., 109, 360 (1991).
*
*  A. Hebert and G. Mathonniere, 'Development of a Third-Generation
*  Superhomogeneisation Method for the Homogenization of a
*  Pressurized Water Reactor Assembly', Nucl. Sci. Eng., 115, 129
*  (1993).
*
*  T. Courau, M. Cometto, E. Girardi, D. Couyras and N. Schwartz,
*  'Elements of Validation of Pin-by-Pin Calculations with the Future
*  EDF Calculation Scheme Based on APOLLO2 and COCAGNE Codes',
*  Proceedings of ICAPP '08 Anaheim, CA USA, June 8-12, 2008.
*
*-----------------------------------------------------------------------
*
      USE GANLIB
*----
*  SUBROUTINE ARGUMENTS
*----
      TYPE(C_PTR) IPTRK2,IPMACR,IPFLX
      INTEGER NBMIX2,IFTRAK,NSPH,KSPH,MAXIT,MAXNBI,IPRINT,IMC,NGCOND,
     1 NMERGE,NALBP,ISCAT,NREG2,NUN2,MAT2(NREG2),KEY2(NREG2),
     2 MERG2(NBMIX2),IGRMIN,IGRMAX
      REAL VOL2(NREG2),SPH(NMERGE+NALBP,NGCOND)
      LOGICAL ILK
      CHARACTER CDOOR*12,CTITRE*72
*----
*  LOCAL VARIABLES
*----
      PARAMETER(NSTATE=40)
      INTEGER IPAR(NSTATE)
      LOGICAL LNORM,LEXAC,LDIFF,REBFLG
      DOUBLE PRECISION FLXTOT,VOLTOT
      CHARACTER TEXT12*12,HSMG*131
      TYPE(C_PTR) IPADF,IPSYS2,JPSYS2,JPFLX,KPSYS2,IPSOU
*----
*  ALLOCATABLE ARRAYS
*----
      INTEGER, ALLOCATABLE, DIMENSION(:) :: NPSYS
      REAL, ALLOCATABLE, DIMENSION(:) :: SIGMD,VOLMER,FACTOR,OUTG1,
     1 OUTG2,COURIN,COUROW,SNORM,SPHNEW
      REAL, ALLOCATABLE, DIMENSION(:,:) :: SIGMA,SIGMS,DIFF,FUNKNO,
     1 SUNKNO,FLXMER,ZLEAK,ALB1,ALB2,WORK
      REAL, ALLOCATABLE, DIMENSION(:,:,:) :: SIGT,SIGW
      REAL, ALLOCATABLE, DIMENSION(:,:,:,:) :: SUNMER
      LOGICAL, ALLOCATABLE, DIMENSION(:) :: LFISS
*----
*  SCRATCH STORAGE ALLOCATION
*----
      ALLOCATE(NPSYS(NGCOND))
      ALLOCATE(SIGMA(0:NBMIX2,ISCAT+1),SIGMD(0:NBMIX2),COURIN(NGCOND),
     1 COUROW(NGCOND),VOLMER(NMERGE),SIGW(NMERGE,NGCOND,ISCAT+1),
     2 FLXMER(NMERGE,NGCOND),DIFF(NMERGE,NGCOND),ZLEAK(NMERGE,NGCOND),
     3 SUNMER(NMERGE,NGCOND,NGCOND,ISCAT),FUNKNO(NUN2,NGCOND),
     4 SUNKNO(NUN2,NGCOND),FACTOR(NMERGE),OUTG1(NGCOND),OUTG2(NGCOND),
     5 SNORM(NGCOND),ALB1(NALBP,NGCOND),ALB2(NALBP,NGCOND),
     6 LFISS(NMERGE))
*----
*  CALCULATION OF THE REFERENCE MERGED/CONDENSED SET OF CROSS SECTIONS
*----
      IF(NALBP.GT.1) CALL XABORT('SPHEQU: NALBP<=1 EXPECTED.')
      CALL LCMGET(IPMACR,'STATE-VECTOR',IPAR)
      NL=IPAR(3)
      NIFISS=IPAR(4)
      ILEAKS=IPAR(9)
      NW=MAX(1,IPAR(10))
      IF(IPAR(2).NE.NMERGE) THEN
        CALL XABORT('SPHEQU: INVALID VALUE OF NMERGE.')
      ELSE IF(IPAR(8).NE.NALBP) THEN
        CALL XABORT('SPHEQU: INVALID VALUE OF NALBP.')
      ENDIF
      ALLOCATE(SIGT(NMERGE,NGCOND,NW+1),SIGMS(0:NBMIX2,NW+1))
      CALL SPHMAC(IPMACR,IPRINT,NMERGE,NALBP,NGCOND,ISCAT,NW,NIFISS,
     1 ILEAKS,VOLMER,FLXMER,SUNMER,SIGT,SIGW,DIFF,ZLEAK,OUTG2,ALB2,
     2 LFISS)
*
      DO 30 INL=1,ISCAT
      DO 20 IGR=1,NGCOND
      DO 10 IKK=1,NMERGE
      SUNMER(IKK,IGR,IGR,INL)=SUNMER(IKK,IGR,IGR,INL)-SIGW(IKK,IGR,INL)
   10 CONTINUE
   20 CONTINUE
   30 CONTINUE
      LDIFF=.FALSE.
      NLF=0
      NANI=0
      IF((NSPH.EQ.4).AND.((CDOOR.EQ.'BIVAC').OR.(CDOOR.EQ.'TRIVAC')))
     > THEN
*        TRANSPORT-DIFFUSION EQUIVALENCE.
         IF(.NOT.C_ASSOCIATED(IPTRK2)) THEN
            CALL XABORT('SPHEQU: MACRO-TRACKING NOT DEFINED(1).')
         ENDIF
         CALL LCMGET(IPTRK2,'STATE-VECTOR',IPAR)
         IF(CDOOR.EQ.'BIVAC') THEN
            NLF=IPAR(14)
            NANI=IPAR(16)
         ELSE IF(CDOOR.EQ.'TRIVAC') THEN
            NLF=IPAR(30)
            NANI=IPAR(32)
         ENDIF
         LDIFF=(NLF.EQ.0).OR.(NANI.LT.0)
         NANI=ABS(NANI)
         IF(NANI.GT.ISCAT) CALL XABORT('SPHEQU: ISCAT OVERFLOW.')
         IF(LDIFF) THEN
           IF(ILEAKS.EQ.0) CALL XABORT('SPHEQU: UNABLE TO COMPUTE DIFF'
     >     //'USION COEFFICIENTS.')
         ENDIF
      ENDIF
      IF(NLF.EQ.0) NANI=1
*----
*  RECOVER THE AVERAGED GAP AND ROW FLUXES FOR EDF-TYPE NORMALIZATION.
*----
      IF(KSPH.GE.2) THEN
         CALL LCMLEN(IPMACR,'ADF',ILONG,ITYLCM)
         IF(ILONG.EQ.0) THEN
            CALL LCMLIB(IPMACR)
            CALL XABORT('SPHEQU: NO ADF DIRECTORY IN THE MACROLIB.')
         ENDIF
         IPADF=LCMDID(IPMACR,'ADF')
      ENDIF
      IF((KSPH.EQ.2).OR.(KSPH.EQ.5)) THEN
         CALL LCMLEN(IPADF,'ALBS00',ILONG,ITYLCM)
         IF(ILONG.NE.2*NGCOND) THEN
           WRITE(HSMG,'(26HSPHEQU: BAD ALBS00 LENGTH=,I5,10H EXPECTED=,
     >     I5,1H.)') ILONG,2*NGCOND
           CALL XABORT(HSMG)
         ENDIF
         ALLOCATE(WORK(NGCOND,2))
         CALL LCMGET(IPADF,'ALBS00',WORK)
         COURIN(:NGCOND)=WORK(:NGCOND,1)
         DEALLOCATE(WORK)
         IF(IPRINT.GT.3) THEN
           WRITE(6,'(/45H SPHEQU: THE VALUES OF ALBS PER MACRO-GROUPS ,
     >     3HARE)')
           WRITE(6,'(1X,1P,10E13.5)') (COURIN(IGR),IGR=1,NGCOND)
         ENDIF
      ELSE IF((KSPH.EQ.3).OR.(KSPH.EQ.6)) THEN
         CALL LCMLEN(IPADF,'FD_B',ILONG,ITYLCM)
         IF(ILONG.NE.NMERGE*NGCOND) THEN
           WRITE(HSMG,'(24HSPHEQU: BAD FD_B LENGTH=,I5,10H EXPECTED=,
     >     I5,5H (1).)') ILONG,NMERGE*NGCOND
           CALL XABORT(HSMG)
         ENDIF
         ALLOCATE(WORK(NMERGE,NGCOND))
         CALL LCMGET(IPADF,'FD_B',WORK)
         DO IGR=1,NGCOND
           VOLTOT=0.0D0
           FLXTOT=0.0D0
           DO IKK=1,NMERGE
             VOLTOT=VOLTOT+VOLMER(IKK)
             FLXTOT=FLXTOT+WORK(IKK,IGR)*VOLMER(IKK)
           ENDDO
           COURIN(IGR)=REAL(FLXTOT/VOLTOT)
         ENDDO
         DEALLOCATE(WORK)
         IF(IPRINT.GT.3) THEN
           WRITE(6,'(/45H SPHEQU: THE VALUES OF FD_B PER MACRO-GROUPS ,
     >     3HARE)')
           WRITE(6,'(1X,1P,10E13.5)') (COURIN(IGR),IGR=1,NGCOND)
         ENDIF
      ELSE IF(KSPH.EQ.4) THEN
         CALL LCMLEN(IPADF,'FD_B',ILONG,ITYLCM)
         IF(ILONG.NE.NMERGE*NGCOND) THEN
           WRITE(HSMG,'(24HSPHEQU: BAD FD_B LENGTH=,I5,10H EXPECTED=,
     >     I5,5H (2).)') ILONG,NMERGE*NGCOND
           CALL XABORT(HSMG)
         ENDIF
         ALLOCATE(WORK(NMERGE,NGCOND))
         CALL LCMGET(IPADF,'FD_B',WORK)
         DO IGR=1,NGCOND
           VOLTOT=0.0D0
           FLXTOT=0.0D0
           DO IKK=1,NMERGE
             VOLTOT=VOLTOT+VOLMER(IKK)
             FLXTOT=FLXTOT+WORK(IKK,IGR)*VOLMER(IKK)
           ENDDO
           COURIN(IGR)=REAL(FLXTOT/VOLTOT)
         ENDDO
         CALL LCMLEN(IPADF,'FD_H',ILONG,ITYLCM)
         IF(ILONG.NE.NMERGE*NGCOND) THEN
           WRITE(HSMG,'(24HSPHEQU: BAD FD_H LENGTH=,I5,10H EXPECTED=,
     >     I5,1H.)') ILONG,NMERGE*NGCOND
           CALL XABORT(HSMG)
         ENDIF
         CALL LCMGET(IPADF,'FD_H',WORK)
         DO IGR=1,NGCOND
           VOLTOT=0.0D0
           FLXTOT=0.0D0
           DO IKK=1,NMERGE
             VOLTOT=VOLTOT+VOLMER(IKK)
             FLXTOT=FLXTOT+WORK(IKK,IGR)*VOLMER(IKK)
           ENDDO
           COUROW(IGR)=REAL(FLXTOT/VOLTOT)
         ENDDO
         DEALLOCATE(WORK)
         IF(IPRINT.GT.3) THEN
           WRITE(6,'(/45H SPHEQU: THE VALUES OF FD_B PER MACRO-GROUPS ,
     >     3HARE)')
           WRITE(6,'(1X,1P,10E13.5)') (COURIN(IGR),IGR=1,NGCOND)
           WRITE(6,'(/45H SPHEQU: THE VALUES OF FD_H PER MACRO-GROUPS ,
     >     3HARE)')
           WRITE(6,'(1X,1P,10E13.5)') (COUROW(IGR),IGR=1,NGCOND)
         ENDIF
      ENDIF
*----
*  ITERATIVE STRATEGY USED TO COMPUTE THE SPH FACTORS
*----
      IPRIN2=MAX(0,IPRINT-5)
      CALL LCMOP(IPSYS2,'SPH$SYS',0,1,0)
      JPSYS2=LCMLID(IPSYS2,'GROUP',NGCOND)
*----
*  REMOVE DB2 LEAKAGE FROM SOURCES
*----
      IF(ILEAKS.NE.0) THEN
         DO 50 IGR=1,NGCOND
         DO 40 IKK=1,NMERGE
         SUNMER(IKK,IGR,IGR,1)=SUNMER(IKK,IGR,IGR,1)-ZLEAK(IKK,IGR)
   40    CONTINUE
   50    CONTINUE
      ENDIF
*----
*  SET SPH NORMALIZATION INFORMATION
*----
      IF(KSPH.LT.0) THEN
*        ASYMTTOTIC NORMALIZATION WITH RESPECT TO MIXTURE -KSPH.
         DO 70 IGR=1,NGCOND
         VOLTOT=0.0D0
         FLXTOT=0.0D0
         DO 60 IKK=1,NMERGE
         IF(IKK.EQ.-KSPH) THEN
            VOLTOT=VOLTOT+VOLMER(IKK)
            FLXTOT=FLXTOT+FLXMER(IKK,IGR)*VOLMER(IKK)
         ENDIF
   60    CONTINUE
         IF(VOLTOT.EQ.0.0) CALL XABORT('SPHEQU: ASYMPTOTIC NORMALIZATI'
     >   //'ON FAILURE.')
         COURIN(IGR)=REAL(FLXTOT/VOLTOT)
         SPH(:NMERGE,IGR)=1.0
   70    CONTINUE
         IF(IPRINT.GT.3) THEN
            WRITE(6,910) (COURIN(IGR),IGR=1,NGCOND)
            WRITE(6,'(/)')
         ENDIF
      ELSE IF(KSPH.EQ.1) THEN
*        AVERAGE FLUX NORMALIZATION.
         DO 90 IGR=1,NGCOND
         VOLTOT=0.0D0
         FLXTOT=0.0D0
         DO 80 IKK=1,NMERGE
         VOLTOT=VOLTOT+VOLMER(IKK)
         FLXTOT=FLXTOT+FLXMER(IKK,IGR)*VOLMER(IKK)
   80    CONTINUE
         COURIN(IGR)=REAL(FLXTOT/VOLTOT)
   90    CONTINUE
         IF(IPRINT.GT.3) THEN
            WRITE(6,910) (COURIN(IGR),IGR=1,NGCOND)
            WRITE(6,'(/)')
         ENDIF
      ELSE IF((KSPH.EQ.2).OR.(KSPH.EQ.3).OR.(KSPH.EQ.5)) THEN
*        SELENGUT NORMALIZATION.
         DO 120 IGR=1,NGCOND
         VOLTOT=0.0D0
         FLXTOT=0.0D0
         DO 100 IKK=1,NMERGE
         VOLTOT=VOLTOT+VOLMER(IKK)
         FLXTOT=FLXTOT+FLXMER(IKK,IGR)*VOLMER(IKK)
  100    CONTINUE
         FLXTOT=FLXTOT/VOLTOT
         DO 110 IKK=1,NMERGE
         SPH(IKK,IGR)=REAL(FLXTOT)/COURIN(IGR)
  110    CONTINUE
  120    CONTINUE
      ELSE IF(KSPH.EQ.4) THEN
*        GENERALIZED SELENGUT NORMALIZATION (EDF-TYPE).
         DO 150 IGR=1,NGCOND
         VOLTOT=0.0D0
         FLXTOT=0.0D0
         DO 130 IKK=1,NMERGE
         VOLTOT=VOLTOT+VOLMER(IKK)
         FLXTOT=FLXTOT+FLXMER(IKK,IGR)*VOLMER(IKK)
  130    CONTINUE
         FLXTOT=FLXTOT/VOLTOT
         DO 140 IKK=1,NMERGE
         SPH(IKK,IGR)=COUROW(IGR)/COURIN(IGR)
  140    CONTINUE
         COURIN(IGR)=COURIN(IGR)*REAL(FLXTOT)/COUROW(IGR)
  150    CONTINUE
      ELSE IF(KSPH.EQ.6) THEN
*        MACRO-CALCULATION WATER GAP NORMALIZATION.
* COUROW = flux in gap in diffusion, initialized to flux in gap in transport
* FUNKNO = FLXMER : flux in diffusion, initialized to flux in transport
         DO 160 IGR=1,NGCOND
         COUROW(IGR)=COURIN(IGR)
         SPH(:NMERGE,IGR)=1.0
  160    CONTINUE
      ELSE IF(KSPH.EQ.7) THEN
*        AVERAGE FLUX NORMALIZATION IN FISSILE ZONES.
         DO 180 IGR=1,NGCOND
         VOLTOT=0.0D0
         FLXTOT=0.0D0
         DO 170 IKK=1,NMERGE
         IF(LFISS(IKK)) THEN
            VOLTOT=VOLTOT+VOLMER(IKK)
            FLXTOT=FLXTOT+FLXMER(IKK,IGR)*VOLMER(IKK)
         ENDIF
  170    CONTINUE
         COURIN(IGR)=REAL(FLXTOT/VOLTOT)
  180    CONTINUE
         IF(IPRINT.GT.3) THEN
            WRITE(6,910) (COURIN(IGR),IGR=1,NGCOND)
            WRITE(6,'(/)')
         ENDIF
      ENDIF
      IF((NSPH.EQ.2).AND.(KSPH.NE.5)) GO TO 470
*----
*  SPH ITERATIONS.
*----
      ITER=0
      IF(C_ASSOCIATED(IPFLX)) THEN
        CALL LCMGET(IPFLX,'STATE-VECTOR',IPAR)
        IF(IPAR(1).NE.NGCOND) CALL XABORT('SPHEQU: INVALID NB OF GROUP'
     1  //'S IN THE INITIALIZATION FLUX.')
        IF(IPAR(2).NE.NUN2) CALL XABORT('SPHEQU: INVALID NB OF UNKNOWN'
     1  //'S IN THE INITIALIZATION FLUX.')
        JPFLX=LCMGID(IPFLX,'FLUX')
        DO 190 IGR=1,NGCOND
          CALL LCMGDL(JPFLX,IGR,FUNKNO(1,IGR))
  190   CONTINUE
        ITER=1
      ELSE IF((CDOOR.EQ.'BIVAC').OR.(CDOOR.EQ.'TRIVAC')) THEN
        DO 200 IGR=1,NGCOND
        FUNKNO(:NUN2,IGR)=1.0
  200   CONTINUE
      ELSE
        DO 220 IGR=1,NGCOND
        FUNKNO(:NUN2,IGR)=0.0
        DO 210 IREG=1,NREG2
        IMAT=MAT2(IREG)
        IF(IMAT.GT.0) FUNKNO(KEY2(IREG),IGR)=FLXMER(MERG2(IMAT),IGR)
  210   CONTINUE
  220   CONTINUE
      ENDIF
      OLDERR=1.0
      NBIERR=0
  230 ITER=ITER+1
      IF(ITER.GE.MAXIT) THEN
        WRITE(6,'(/46H SPHEQU: MAX. NUMBER OF ITERATIONS IS REACHED.)')
        GO TO 440
      ENDIF
      ERROR=0.0
      ERR2=0.0
      DO 240 IREG=1,NREG2
      IF(MAT2(IREG).GT.NBMIX2) THEN
         CALL XABORT('SPHEQU: INVALID MACRO-MIXTURE INDEX.')
      ENDIF
  240 CONTINUE
*----
*  SET MACROSCOPIC CROSS SECTIONS IN THE IPSYS2 LCM OBJECT.
*----
      NPSYS(:NGCOND)=0
      DO 310 IGR=IGRMIN,IGRMAX
      SIGMS(0:NBMIX2,:NW+1)=0.0
      SIGMA(0:NBMIX2,:ISCAT+1)=0.0
      SIGMD(0:NBMIX2)=0.0
      DO 280 IREG=1,NREG2
      IMAT=MAT2(IREG)
      IF(IMAT.EQ.0) GO TO 280
      IMERG=MERG2(IMAT)
      IF(LDIFF) SIGMD(IMAT)=DIFF(IMERG,IGR)*SPH(IMERG,IGR)
      IF(IMC.EQ.1) THEN
        SIGMA(IMAT,1)=SIGW(IMERG,IGR,1)*SPH(IMERG,IGR)
        DO 250 IW=1,NW+1
        IF(MOD(IW-1,2).EQ.0) THEN
          SIGMS(IMAT,IW)=SIGT(IMERG,IGR,IW)*SPH(IMERG,IGR)
        ELSE IF(MOD(IW-1,2).EQ.1) THEN
          SIGMS(IMAT,IW)=SIGT(IMERG,IGR,IW)/SPH(IMERG,IGR)
        ENDIF
  250   CONTINUE
      ELSE IF(IMC.EQ.2) THEN
*       TRANSPORT-PIJ EQUIVALENCE WITHOUT BELL FACTOR ACCELERATION.
        SIGMA(IMAT,1)=SIGW(IMERG,IGR,1)*SPH(IMERG,IGR)+SIGT(IMERG,IGR,1)
     >  *(1.0-SPH(IMERG,IGR))
        DO 260 IW=1,NW+1
        SIGMS(IMAT,IW)=SIGT(IMERG,IGR,IW)
  260   CONTINUE
      ELSE IF(IMC.EQ.3) THEN
*       TRANSPORT-PIJ EQUIVALENCE WITH BELL FACTOR ACCELERATION.
        SIGMA(IMAT,1)=0.0
        DO 270 IW=1,NW+1
        SIGMS(IMAT,IW)=SIGT(IMERG,IGR,IW)
  270   CONTINUE
      ENDIF
  280 CONTINUE
      NPSYS(IGR)=IGR
      KPSYS2=LCMDIL(JPSYS2,IGR)
      DO 290 IW=1,MIN(NW+1,10)
      IF(IW.EQ.1) THEN
        TEXT12='DRAGON-TXSC'
      ELSE
        WRITE(TEXT12,'(8HDRAGON-T,I1,3HXSC)') IW-1
      ENDIF
      CALL LCMPUT(KPSYS2,TEXT12,NBMIX2+1,2,SIGMS(0,IW))
  290 CONTINUE
      CALL LCMPUT(KPSYS2,'DRAGON-S0XSC',NBMIX2+1,2,SIGMA(0,1))
      IF(LDIFF) THEN
         SIGMD(0)=1.0E10
         CALL LCMPUT(KPSYS2,'DRAGON-DIFF',NBMIX2+1,2,SIGMD(0))
      ENDIF
*----
*  SPH CORRECTION OF PHYSICAL ALBEDOS
*----
      IF(NALBP.GT.0) THEN
         DO 300 IAL=1,NALBP
         FACT=0.5*(1.0-ALB2(IAL,IGR))/(1.0+ALB2(IAL,IGR))*
     1   SPH(NMERGE+IAL,IGR)
         ALB1(IAL,IGR)=(1.0-2.0*FACT)/(1.0+2.0*FACT)
  300    CONTINUE
         CALL LCMPUT(KPSYS2,'ALBEDO',NALBP,2,ALB1(1,IGR))
      ENDIF
  310 CONTINUE
*----
*  ASSEMBLY OF PIJ OR SYSTEM MATRICES AT ITERATION ITER.
*----
      IF(.NOT.C_ASSOCIATED(IPTRK2)) THEN
         CALL XABORT('SPHEQU: MACRO-TRACKING NOT DEFINED(2).')
      ENDIF
      ISTRM=1
      KNORM=1
      IPHASE=2
      IF(NSPH.EQ.4) IPHASE=1
      IF(IPHASE.EQ.2) THEN
         IPIJK=1
         ITPIJ=1
         LNORM=.FALSE.
         CALL DOORPV(CDOOR,JPSYS2,NPSYS,IPTRK2,IFTRAK,IPRIN2,NGCOND,
     >   NREG2,NBMIX2,NANI,MAT2,VOL2,KNORM,IPIJK,ILK,ITPIJ,LNORM,
     >   CTITRE,NALBP)
      ELSE
         CALL DOORAV(CDOOR,JPSYS2,NPSYS,IPTRK2,IFTRAK,IPRIN2,NGCOND,
     >   NREG2,NBMIX2,NANI,NW,MAT2,VOL2,KNORM,ILK,CTITRE,NALBP,ISTRM)
      ENDIF
*----
*  MACRO-FLUX CALCULATION AT ITERATION ITER.
*----
      IF((NSPH.EQ.4).AND.(CDOOR.EQ.'BIVAC')) THEN
         CALL SPHBIV(IPTRK2,JPSYS2,NPSYS,NREG2,NUN2,NMERGE,NALBP,NGCOND,
     1   NBMIX2,MAT2,VOL2,KEY2,MERG2,SUNMER(1,1,1,1),FLXMER,SPH,ITER,
     2   FUNKNO,SUNKNO)
      ELSE IF((NSPH.EQ.4).AND.(CDOOR.EQ.'TRIVAC')) THEN
         CALL SPHTRI(IPTRK2,JPSYS2,NPSYS,NREG2,NUN2,NMERGE,NALBP,NGCOND,
     1   NBMIX2,MAT2,VOL2,KEY2,MERG2,SUNMER(1,1,1,1),FLXMER,SPH,ITER,
     2   FUNKNO,SUNKNO)
      ELSE IF((NSPH.EQ.4).AND.(CDOOR.EQ.'SN')) THEN
*        TRANSPORT-SN EQUIVALENCE.
         CALL SPHSN(IPTRK2,NPSYS,NREG2,NUN2,NMERGE,NALBP,NGCOND,NBMIX2,
     1   MAT2,VOL2,KEY2,MERG2,SUNMER(1,1,1,1),FLXMER,SPH,ITER,FUNKNO,
     2   SUNKNO)
      ELSE IF(IMC.LE.2) THEN
*        TRANSPORT-PIJ EQUIVALENCE.
         CALL SPHPIJ(NPSYS,NREG2,NUN2,NMERGE,NALBP,NGCOND,NBMIX2,MAT2,
     1   VOL2,KEY2,MERG2,SUNMER(1,1,1,1),FLXMER,SPH,ITER,FUNKNO,SUNKNO)
      ELSE IF(IMC.EQ.3) THEN
*        TRANSPORT-PIJ EQUIVALENCE WITH BELL FACTOR ACCELERATION.
         CALL SPHTRA(JPSYS2,ITER,NPSYS,KSPH,NREG2,NUN2,NMERGE,NALBP,
     1   NGCOND,SUNMER(1,1,1,1),FLXMER,NBMIX2,MAT2,VOL2,KEY2,MERG2,SPH,
     2   SIGW(1,1,1),SIGT(1,1,1),COURIN,FUNKNO)
         SNORM(:NGCOND)=1.0
         GO TO 390
      ENDIF
*----
*  COMPUTE THE MACRO-FLUX USING THE VECTORIAL DOOR.
*----
      IDIR=0
      IPSOU=C_NULL_PTR
      LEXAC=.FALSE.
      REBFLG=.FALSE.
      CALL DOORFV(CDOOR,JPSYS2,NPSYS,IPTRK2,IFTRAK,IPRIN2,NGCOND,
     1 NBMIX2,IDIR,NREG2,NUN2,IPHASE,LEXAC,MAT2,VOL2,KEY2,CTITRE,
     2 SUNKNO,FUNKNO,IPMACR,IPSOU,REBFLG)
*----
*  COMPUTE MACRO-CALCULATION LEAKAGE RATES IF NALBP.GT.0
*----
      IF(NALBP.GT.0) THEN
         DO 340 IGR=1,NGCOND
         OUTG1(IGR)=0.0
         DO 330 K=1,NREG2
         L=MAT2(K)
         IF(L.EQ.0) GO TO 330
         IUN=KEY2(K)
         IF(VOL2(K).EQ.0.0) GO TO 330
         IKK=MERG2(L)
         OUTG1(IGR)=OUTG1(IGR)+(SIGW(IKK,IGR,1)-SIGT(IKK,IGR,1))*
     >   FUNKNO(IUN,IGR)*VOLMER(IKK)*SPH(IKK,IGR)
         DO 320 JGR=1,NGCOND
         OUTG1(IGR)=OUTG1(IGR)+SUNMER(IKK,JGR,IGR,1)*FUNKNO(IUN,JGR)*
     >   VOLMER(IKK)*SPH(IKK,JGR)
  320    CONTINUE
  330    CONTINUE
         IF(IPRIN2.GT.0) WRITE(6,920) IGR,OUTG1(IGR)
  340    CONTINUE
      ENDIF
*----
*  MACRO-FLUX NORMALIZATION.
*----
      IF(ILK.AND.(NALBP.EQ.0)) GO TO 390
      SNORM(:NGCOND)=1.0
      IF(KSPH.LT.0) THEN
*        ASYMTTOTIC NORMALIZATION WITH RESPECT TO MIXTURE -KSPH.
         IF(-KSPH.GT.NMERGE) CALL XABORT('SPHEQU: INVALID ASYMPTOTIC M'
     >   //'IXTURE SET.')
         DO 360 IGR=1,NGCOND
         IF(NPSYS(IGR).EQ.0) GO TO 360
         VOLTOT=0.0D0
         FLXTOT=0.0D0
         DO 350 IREG=1,NREG2
         IMAT=MAT2(IREG)
         IF(IMAT.EQ.0) GO TO 350
         IF(MERG2(IMAT).EQ.-KSPH) THEN
            VOLTOT=VOLTOT+VOL2(IREG)
            FLXTOT=FLXTOT+FUNKNO(KEY2(IREG),IGR)*VOL2(IREG)
         ENDIF
  350    CONTINUE
         IF(VOLTOT.GT.0) THEN
            FLXTOT=FLXTOT/VOLTOT
            FFF=COURIN(IGR)/REAL(FLXTOT)
         ELSE
            FFF=1.0
         ENDIF
         SNORM(IGR)=FFF
         IF(IPRIN2.GT.0) WRITE(6,960) IGR,FFF
  360    CONTINUE
      ELSE
*        AVERAGE OR SELENGUT FLUX NORMALIZATION.
         DO 380 IGR=1,NGCOND
         IF(NPSYS(IGR).EQ.0) GO TO 380
         VOLTOT=0.0D0
         FLXTOT=0.0D0
         DO 370 IREG=1,NREG2
         IMAT=MAT2(IREG)
         IF(IMAT.EQ.0) GO TO 370
         IMERG=MERG2(IMAT)
         IF((KSPH.NE.7).OR.LFISS(IMERG)) THEN
            VOLTOT=VOLTOT+VOL2(IREG)
            FLXTOT=FLXTOT+FUNKNO(KEY2(IREG),IGR)*VOL2(IREG)
         ENDIF
  370    CONTINUE
         FLXTOT=FLXTOT/VOLTOT
         IF(IPRINT.GE.100) THEN
           WRITE(6,*)'FLXTOT: =',FLXTOT,'VOLTOT: =',VOLTOT
         ENDIF
         IF(KSPH.NE.6) THEN
           SNORM(IGR)=COURIN(IGR)/REAL(FLXTOT)
         ELSE
*          Compute diffusion flux in water gap => COUROW(IGR)
           IF(CDOOR.NE.'TRIVAC') CALL XABORT('SPHEQU: TRIVAC expected'
     1     //' as tracking module with SELE-GAP')
           CALL SPHGAP(IPTRK2,IPRINT,NREG2,NUN2,MAT2,KEY2,FUNKNO(1,IGR),
     1     COUROW(IGR))
           IF(IPRINT.GE.100) THEN
             WRITE(6,*)'COURIN: =',COURIN(IGR),'COUROW: =',COUROW(IGR)
           ENDIF
           SNORM(IGR)=COURIN(IGR)/COUROW(IGR)
         ENDIF
         IF(IPRIN2.GT.0) WRITE(6,960) IGR,SNORM(IGR)
  380    CONTINUE
      ENDIF
*----
*  COMPUTE THE IMPROVED SPH FACTORS.
*----
  390 ALLOCATE(SPHNEW(NMERGE+NALBP))
      DO 430 IGR=1,NGCOND
      IF(NPSYS(IGR).EQ.0) GO TO 430
      VOLMER(:NMERGE)=0.0
      FACTOR(:NMERGE)=0.0
      SPHNEW(:NMERGE+NALBP)=1.0
      DO 400 IREG=1,NREG2
      IMAT=MAT2(IREG)
      IF(IMAT.EQ.0) GO TO 400
      IKK=MERG2(IMAT)
      VOLMER(IKK)=VOLMER(IKK)+VOL2(IREG)
      FACTOR(IKK)=FACTOR(IKK)+FUNKNO(KEY2(IREG),IGR)*VOL2(IREG)
  400 CONTINUE
      DO 410 IKK=1,NMERGE
      IF(VOLMER(IKK).EQ.0.0) GO TO 410
      FACTOR(IKK)=FACTOR(IKK)/VOLMER(IKK)
      SPHNEW(IKK)=FLXMER(IKK,IGR)/(SNORM(IGR)*FACTOR(IKK))
      IF(SPHNEW(IKK).LT.0.0) THEN
         WRITE(6,980) IGR,IKK
         SPHNEW(IKK)=1.0
      ENDIF
  410 CONTINUE
      IF(NALBP.EQ.1) SPHNEW(NMERGE+1)=OUTG1(IGR)/(OUTG2(IGR)*SNORM(IGR))
      DO 420 IKK=1,NMERGE+NALBP
      ERRT=ABS((SPHNEW(IKK)-SPH(IKK,IGR))/SPHNEW(IKK))
      ERR2=ERR2+ERRT*ERRT
      ERROR=MAX(ERROR,ERRT)
      SPH(IKK,IGR)=SPHNEW(IKK)
  420 CONTINUE
      IF(IPRINT.GT.4) THEN
         WRITE(6,930) 'NSPH',IGR,(SPH(IKK,IGR),IKK=1,NMERGE+NALBP)
      ENDIF
      IF(IPRINT.GT.5) THEN
         WRITE(6,930) 'FUNKNO',IGR,(FUNKNO(IUNK,IGR),IUNK=1,NUN2)
      ENDIF
  430 CONTINUE
      DEALLOCATE(SPHNEW)
      ERR2=SQRT(ERR2/(NMERGE*NGCOND))
      IF(IPRINT.GT.1) WRITE(6,935) ITER,ERROR,ERR2
      IF(IPRINT.GT.2) THEN
         IF(ERROR.GE.EPSPH) WRITE(6,940) ((IKK,IGR,SPH(IKK,IGR),
     >   IKK=1,NMERGE+NALBP),IGR=1,NGCOND)
      ENDIF
      IF(ERR2.LT.EPSPH) GO TO 440
      IF((ITER.GT.1).AND.(ERR2.GT.OLDERR)) THEN
         WRITE(6,970) ITER
         NBIERR=NBIERR+1
         IF(NBIERR.GE.MAXNBI) THEN
            WRITE(6,990) ITER
            GO TO 440
         ENDIF
      ENDIF
      OLDERR=ERR2
      GO TO 230
  440 WRITE(6,950) ITER
*----
*  RESET SOURCES TO NO DB2 LEAKAGE
*----
      IF(ILEAKS.NE.0) THEN
         DO 460 IGR=1,NGCOND
         DO 450 IKK=1,NMERGE
         SUNMER(IKK,IGR,IGR,1)=SUNMER(IKK,IGR,IGR,1)+ZLEAK(IKK,IGR)
  450    CONTINUE
  460    CONTINUE
      ENDIF
      CALL LCMCL(IPSYS2,2)
*----
*  SCRATCH STORAGE DEALLOCATION
*----
  470 DEALLOCATE(LFISS,ALB2,ALB1,SNORM,OUTG2,OUTG1,FACTOR,SUNKNO,
     1 FUNKNO,SUNMER,ZLEAK,DIFF,FLXMER,SIGW,SIGT,VOLMER,COUROW,COURIN,
     2 SIGMD,SIGMS,SIGMA)
      DEALLOCATE(NPSYS)
      RETURN
*
  910 FORMAT(/44H SPHEQU: AVERAGE FLUXES PER MACRO-GROUPS ARE/
     > (1X,1P,10E13.5))
  920 FORMAT(/8H SPHEQU:,5X,6HGROUP=,I4,15H MACRO LEAKAGE=,1P,E12.4)
  930 FORMAT(/26H SPHEQU: VALUES OF VECTOR ,A,9H IN GROUP,I5,4H ARE/
     > (1X,1P,10E13.5))
  935 FORMAT(/14H SPHEQU: ITER=,I3,4X,6HERROR=,1P,E10.3,1X,6HERR 2=,
     > E10.3)
  940 FORMAT(4X,4HSPH(,I3,1H,,I3,2H)=,F9.5,:,4X,4HSPH(,I3,1H,,I3,2H)=,
     > F9.5,:,4X,4HSPH(,I3,1H,,I3,2H)=,F9.5,:,4X,4HSPH(,I3,1H,,I3,2H)=,
     > F9.5,:,4X,4HSPH(,I3,1H,,I3,2H)=,F9.5)
  950 FORMAT(/40H SPHEQU: CONVERGENCE OF SPH ALGORITHM IN,I5,
     > 12H ITERATIONS.)
  960 FORMAT(/43H SPHEQU: FLUX NORMALIZATION FACTOR IN GROUP,I4,1H=,1P,
     1 E13.5)
  970 FORMAT(1X,'Warning: oscillations in SPH at iteration ',i10)
  980 FORMAT(1X,'Warning: negative SPH factor in group ',i5,
     > ' and region ',i5,' set to 1.0')
  990 FORMAT(1X,'Warning: maximum of 3 error oscillations ',
     >'in SPH convergence reached at iteration ',i10)
      END
