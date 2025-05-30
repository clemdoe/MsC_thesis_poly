*DECK DRENOU
      SUBROUTINE DRENOU(IPRINT,IPGPT,IPMAC1,IPMAC2,IPFLX,IPTRK,IPGRAD,
     1 NG,NREG,ITYPE,IELEM,NMIL,NALBP,NUN,NFIS1,NFIS2,ILEAK1,ILEAK2,
     2 IDF2,MATCOD,KEYFLX,VOL,LNO,NFUNC,RMSD)
*
*-----------------------------------------------------------------------
*
*Purpose:
* Compute the GPT sources corresponding to the gradient of the RMS
* absorption distribution. Case with NFUNC individual components to be
* used with a Newtonian method.
*
*Copyright:
* Copyright (C) 2019 Ecole Polytechnique de Montreal
* This library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 2.1 of the License, or (at your option) any later version
*
*Author(s): 
* A. Hebert
*
*Parameters: input
* IPRINT  print parameter
* IPGPT   pointer to the L_GPT data structure.
* IPMAC1  pointer to the actual macrolib structure.
* IPMAC2  pointer to the reference macrolib structure.
* IPFLX   pointer to the multigroup flux.
* IPTRK   pointer to the tracking object.
* IPGRAD  pointer to the L_OPTIMIZE object.
* NG      number of energy groups.
* NREG    number of regions.
* NMIL    number of material mixtures.
* NALBP   number of physical albedos.
* NUN     number of unknowns per energy group.
* NFIS1   number of fissile isotopes in actual macrolib.
* NFIS2   number of fissile isotopes in reference macrolib.
* ILEAK1  type of leakage calculation in actual macrolib
*         =0: no leakage; =1: homogeneous leakage (Diffon).
* ILEAK2  type of leakage calculation in reference macrolib.
* IDF2    ADF type, 0 = none, 1 = Albedo, 2 = FD_B/FD_C/..., 3 = ADF.
* MATCOD  material mixture indices per region.
* KEYFLX  position of averaged fluxes in unknown vector.
* VOL     volumes.
* LNO     flag set to .true. to exit after calculation of RMS.
*
*Parameters: output
* NFUNC   number of individual components in the gradient terms.
* RMSD    RMS error on rate distribution.
*
*Parameters: 
* ITYPE
* IELEM
*
* Reference:
*  A. Hebert,"Developpement de la methode SPH: Homogeneisation de
*  cellules dans un reseau non uniforme et calcul des parametres de
*  reflecteur," Note CEA-N-2209, Sect. 3.5.1, 1981.
*
*-----------------------------------------------------------------------
*
      USE GANLIB
*----
*  SUBROUTINE ARGUMENTS
*----
      TYPE(C_PTR) IPGPT,IPMAC1,IPMAC2,IPFLX,IPTRK,IPGRAD
      INTEGER IPRINT,NG,NREG,ITYPE,IELEM,NMIL,NALBP,NUN,NFIS1,NFIS2,
     > ILEAK1,ILEAK2,IDF2,MATCOD(NREG),KEYFLX(NREG),NFUNC
      REAL VOL(NREG)
      DOUBLE PRECISION RMSD
      LOGICAL LNO
*----
*  LOCAL VARIABLES
*----
      PARAMETER (NSTATE=40)
      TYPE(C_PTR) JPMAC1,JPMAC2,KPMAC1,KPMAC2,JPFLX,JPGPT,KPGPT
      INTEGER ISTATE(NSTATE)
      DOUBLE PRECISION SOUT1,SOUT2,SOUTOT,AB1TOT,AB2TOT,FI1TOT,FI2TOT,
     > SUM1,DSUM,DELTA,OUT,SA,SF,ABS2M,OUT2M,AIL,BIL,DEN1,DEN2
      CHARACTER HSMG*131
      DOUBLE PRECISION, PARAMETER :: EPS=1.0E-4,EPSL=1.0E-4
*----
*  ALLOCATABLE ARRAYS
*----
      INTEGER, ALLOCATABLE, DIMENSION(:) :: IJJ,NJJ,IPOS,IREL,KN,IQFR
      REAL, ALLOCATABLE, DIMENSION(:) :: GAR,WORK,SUNK,FLUX,QFR,OUTG1,
     > OUTG2,DIFHOM,DIFF
      REAL, ALLOCATABLE, DIMENSION(:,:) :: PHI1,PHI2,ABS1,ABS2,NUF1,
     > NUF2,GAMMA,OUTG2R
      REAL, ALLOCATABLE, DIMENSION(:,:,:) :: CHI1,CHI2,RHS1,LHS1,RHS2,
     > LHS2
      DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:) :: VARV,RHS,CONST,FF
      DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:) :: SIGA,SIGF,DFF
*----
*  SCRATCH STORAGE ALLOCATION
*----
      ALLOCATE(PHI1(NMIL,NG),PHI2(NMIL,NG),ABS1(NMIL,NG),ABS2(NMIL,NG),
     1 RHS1(NMIL,NG,NG),LHS1(NMIL,NG,NG),RHS2(NMIL,NG,NG),
     2 LHS2(NMIL,NG,NG),CONST(NG),IREL(NG),RHS(NG),GAMMA(NUN,NG),
     3 OUTG1(NG),OUTG2(NG),OUTG2R(NG,2),SIGA(NMIL,NG),SIGF(NMIL,NG))
*----
*  COMPUTE THE ACTUAL AND REFERENCE REACTION RATE MATRICES
*----
      CALL LCMGET(IPMAC1,'K-EFFECTIVE',ZKEFF1)
      CALL LCMGET(IPMAC2,'K-EFFECTIVE',ZKEFF2)
      IF(IDF2.EQ.1) THEN
        CALL LCMSIX(IPMAC2,'ADF',1)
        CALL LCMLEN(IPMAC2,'ALBS00',ILCMLN,ITYLCM)
        IF(ILCMLN.NE.2*NG) CALL XABORT('DRENOU: WRONG ALBS00 LENGTH.')
        CALL LCMGET(IPMAC2,'ALBS00',OUTG2R)
        CALL LCMSIX(IPMAC2,' ',2)
      ENDIF
      CALL LCMLEN(IPMAC1,'B2  B1HOM',ILCMLN,ITYLCM)
      IF(ILCMLN.EQ.1) THEN
        CALL LCMGET(IPMAC1,'B2  B1HOM',B21)
      ELSE
        B21=0.0
      ENDIF
      CALL LCMLEN(IPMAC2,'B2  B1HOM',ILCMLN,ITYLCM)
      IF(ILCMLN.EQ.1) THEN
        CALL LCMGET(IPMAC2,'B2  B1HOM',B22)
      ELSE
        B22=0.0
      ENDIF
      IF((ILEAK1.EQ.1).AND.(IPRINT.GT.0)) THEN
        WRITE(6,'(/22H DRENOU:     MACRO B2=,1P,E12.4)') B21
      ENDIF
      IF((ILEAK2.EQ.1).AND.(IPRINT.GT.0)) THEN
        WRITE(6,'(/22H DRENOU: REFERENCE B2=,1P,E12.4)') B22
      ENDIF
      RHS1(:NMIL,:NG,:NG)=0.0
      LHS1(:NMIL,:NG,:NG)=0.0
      RHS2(:NMIL,:NG,:NG)=0.0
      LHS2(:NMIL,:NG,:NG)=0.0
      SIGA(:NMIL,:NG)=0.0D0
      SIGF(:NMIL,:NG)=0.0D0
      JPMAC1=LCMGID(IPMAC1,'GROUP')
      JPMAC2=LCMGID(IPMAC2,'GROUP')
      ALLOCATE(IJJ(NMIL),NJJ(NMIL),IPOS(NMIL),GAR(NMIL),WORK(NMIL*NG),
     1 CHI1(NMIL,NFIS1,NG),NUF1(NMIL,NFIS1),CHI2(NMIL,NFIS2,NG),
     2 NUF2(NMIL,NFIS2),DIFHOM(NG),DIFF(NMIL))
      DO IG=1,NG
        KPMAC1=LCMGIL(JPMAC1,IG)
        CALL LCMGET(KPMAC1,'CHI',CHI1(1,1,IG))
        KPMAC2=LCMGIL(JPMAC2,IG)
        CALL LCMGET(KPMAC2,'CHI',CHI2(1,1,IG))
        CALL LCMLEN(KPMAC1,'FLUX-INTG',ILG,ITYLCM)
        IF(ILG.NE.NMIL) CALL XABORT('DRENOU: MISSING ACTUAL FLUX.')
        CALL LCMLEN(KPMAC2,'FLUX-INTG',ILG,ITYLCM)
        IF(ILG.NE.NMIL) CALL XABORT('DRENOU: MISSING REFERENCE FLUX.')
        CALL LCMGET(KPMAC1,'FLUX-INTG',PHI1(1,IG))
        CALL LCMGET(KPMAC2,'FLUX-INTG',PHI2(1,IG))
      ENDDO
      DO IG=1,NG
        KPMAC1=LCMGIL(JPMAC1,IG)
        KPMAC2=LCMGIL(JPMAC2,IG)
        IF(ILEAK1.EQ.1) THEN
          CALL LCMLEN(KPMAC1,'DIFF',ILCMLN,ITYLCM)
          IF(ILCMLN.GT.0) THEN
            CALL LCMGET(KPMAC1,'DIFF',DIFF)
          ELSE
            CALL LCMGET(IPMAC1,'DIFHOMB1HOM',DIFHOM)
            DO IBM=1,NMIL
              DIFF(IBM)=DIFHOM(IG)
            ENDDO
          ENDIF
        ELSE
          DIFF(:NMIL)=0.0
        ENDIF
        CALL LCMGET(KPMAC1,'NTOT0',GAR)
        CALL LCMGET(KPMAC1,'SCAT00',WORK)
        CALL LCMGET(KPMAC1,'NJJS00',NJJ)
        CALL LCMGET(KPMAC1,'IJJS00',IJJ)
        CALL LCMGET(KPMAC1,'IPOS00',IPOS)
        DO IBM=1,NMIL
          SIGA(IBM,IG)=SIGA(IBM,IG)+GAR(IBM)
          IPOSDE=IPOS(IBM)
          DO JG=IJJ(IBM),IJJ(IBM)-NJJ(IBM)+1,-1
*           IG <-- JG
            RHS1(IBM,IG,JG)=RHS1(IBM,IG,JG)-WORK(IPOSDE)*PHI1(IBM,JG)
            SIGA(IBM,JG)=SIGA(IBM,JG)-WORK(IPOSDE)
            IPOSDE=IPOSDE+1
          ENDDO
          RHS1(IBM,IG,IG)=RHS1(IBM,IG,IG)+(GAR(IBM)+B21*DIFF(IBM))*
     >    PHI1(IBM,IG)
        ENDDO
        CALL LCMGET(KPMAC1,'NUSIGF',NUF1)
        DO IBM=1,NMIL
          DO IFIS=1,NFIS1
            DO JG=1,NG
              LHS1(IBM,JG,IG)=LHS1(IBM,JG,IG)+CHI1(IBM,IFIS,JG)*
     >        NUF1(IBM,IFIS)*PHI1(IBM,IG)
              SIGF(IBM,IG)=SIGF(IBM,IG)+CHI1(IBM,IFIS,JG)*NUF1(IBM,IFIS)
            ENDDO
          ENDDO
        ENDDO
*
        IF(ILEAK2.EQ.1) THEN
          CALL LCMLEN(KPMAC2,'DIFF',ILCMLN,ITYLCM)
          IF(ILCMLN.GT.0) THEN
            CALL LCMGET(KPMAC2,'DIFF',DIFF)
          ELSE
            CALL LCMGET(IPMAC2,'DIFHOMB1HOM',DIFHOM)
            DO IBM=1,NMIL
              DIFF(IBM)=DIFHOM(IG)
            ENDDO
          ENDIF
        ELSE
          DIFF(:NMIL)=0.0
        ENDIF
        CALL LCMGET(KPMAC2,'NTOT0',GAR)
        CALL LCMGET(KPMAC2,'SCAT00',WORK)
        CALL LCMGET(KPMAC2,'NJJS00',NJJ)
        CALL LCMGET(KPMAC2,'IJJS00',IJJ)
        CALL LCMGET(KPMAC2,'IPOS00',IPOS)
        DO IBM=1,NMIL
          IPOSDE=IPOS(IBM)
          DO JG=IJJ(IBM),IJJ(IBM)-NJJ(IBM)+1,-1
*           IG <-- JG
            RHS2(IBM,IG,JG)=RHS2(IBM,IG,JG)-WORK(IPOSDE)*PHI2(IBM,JG)
            IPOSDE=IPOSDE+1
          ENDDO
          RHS2(IBM,IG,IG)=RHS2(IBM,IG,IG)+(GAR(IBM)+B22*DIFF(IBM))*
     >    PHI2(IBM,IG)
        ENDDO
        CALL LCMGET(KPMAC2,'NUSIGF',NUF2)
        DO IBM=1,NMIL
          DO IFIS=1,NFIS2
            DO JG=1,NG
              LHS2(IBM,JG,IG)=LHS2(IBM,JG,IG)+CHI2(IBM,IFIS,JG)*
     >        NUF2(IBM,IFIS)*PHI2(IBM,IG)
            ENDDO
          ENDDO
        ENDDO
      ENDDO
      DEALLOCATE(DIFF,DIFHOM,NUF2,CHI2,NUF1,CHI1,WORK,GAR,IPOS,NJJ,IJJ)
*----
*  COMPUTE THE ACTUAL AND REFERENCE ABSORPTION AND FISSION RATES
*----
      AB1TOT=0.0D0
      AB2TOT=0.0D0
      FI1TOT=0.0D0
      FI2TOT=0.0D0
      DO IG=1,NG
        OUTG1(IG)=0.0
        OUTG2(IG)=0.0
        DO IBM=1,NMIL
          OUTG1(IG)=OUTG1(IG)+SUM(LHS1(IBM,IG,:NG))/ZKEFF1-
     1    SUM(RHS1(IBM,IG,:NG))
          OUTG2(IG)=OUTG2(IG)+SUM(LHS2(IBM,IG,:NG))/ZKEFF2-
     1    SUM(RHS2(IBM,IG,:NG))
          ABS1(IBM,IG)=SUM(RHS1(IBM,:NG,IG))
          ABS2(IBM,IG)=SUM(RHS2(IBM,:NG,IG))
          AB1TOT=AB1TOT+ABS1(IBM,IG)
          AB2TOT=AB2TOT+ABS2(IBM,IG)
          FI1TOT=FI1TOT+SUM(LHS1(IBM,:NG,IG))
          FI2TOT=FI2TOT+SUM(LHS2(IBM,:NG,IG))
        ENDDO
        IF(IDF2.GT.0) OUTG2(IG)=OUTG2R(IG,1)-OUTG2R(IG,2)
        IF((NALBP.GT.0).AND.(OUTG2(IG).LT.-1.0E-6)) THEN
          WRITE(HSMG,'(44HDRENOU: INCONSISTENT REFERENCE LEAKAGE IN GR,
     1    3HOUP,I4,7H. LEAK=,1P,E13.4)') IG,OUTG2(IG)
          CALL XABORT(HSMG)
        ENDIF
      ENDDO
*----
*  COMPUTE THE ACTUAL LEAKAGE FROM OUT-CURRENTS
*----
      OUT=0.0D0
      GAMMA(:NUN,:NG)=0.0
      IF(NALBP.GT.0) THEN
        CALL LCMLEN(IPTRK,'KN',MAXKN,ITYLCM)
        CALL LCMLEN(IPTRK,'QFR',MAXQF,ITYLCM)
        ALLOCATE(KN(MAXKN),QFR(MAXQF),IQFR(MAXQF),FLUX(NUN))
        CALL LCMGET(IPTRK,'KN',KN)
        CALL LCMGET(IPTRK,'QFR',QFR)
        CALL LCMGET(IPTRK,'IQFR',IQFR)
        JPFLX=LCMGID(IPFLX,'FLUX')
        DO IG=1,NG
          CALL LCMGDL(JPFLX,IG,FLUX)
          CALL DREJ02(ITYPE,IELEM,NREG,NUN,MAXKN,MAXQF,MATCOD,KN,QFR,
     1    IQFR,VOL,FLUX,OUTG1(IG),GAMMA(1,IG))
          OUT=OUT+OUTG1(IG)
          IF(IPRINT.GT.0) THEN
           WRITE(6,130) IG,OUTG1(IG)/REAL(AB1TOT),OUTG2(IG)/REAL(AB2TOT)
          ENDIF
        ENDDO
        DEALLOCATE(FLUX,IQFR,QFR,KN)
      ENDIF
*----
*  COMPUTE MACRO AND REFERENCE K-EFFECTIVE
*----
      DEN1=0.0D0
      DEN2=0.0D0
      DO IG=1,NG
        OUTG1(IG)=OUTG1(IG)+SUM(ABS1(:NMIL,IG))
        OUTG2(IG)=OUTG2(IG)+SUM(ABS2(:NMIL,IG))
        DEN1=DEN1+OUTG1(IG)
        DEN2=DEN2+OUTG2(IG)
      ENDDO
      IF(IPRINT.GT.0) THEN
        WRITE(6,'(/24H DRENOU:     MACRO KEFF=,1P,E12.5)') FI1TOT/DEN1
        WRITE(6,'(/24H DRENOU: REFERENCE KEFF=,1P,E12.5)') FI2TOT/DEN2
      ENDIF
*----
*  GET INFORMATION FROM L_OPTIMIZE OBJECT
*----
      CALL LCMGET(IPGRAD,'DEL-STATE',ISTATE)
      IF(ISTATE(4).LE.2) CALL XABORT('DRENOU: NO DIRECT EFFECT WITH '
     > //'THIS TYPE OF PERTURBATION.')
      IF(ISTATE(7).NE.1) CALL XABORT('DRENOU: IBM1=1 EXPECTED.')
      IF(ISTATE(8).NE.NMIL) CALL XABORT('DRENOU: IBM2=NMIL EXPECTED.')
      NGR1=ISTATE(5)
      NGR2=ISTATE(6)
      NPERT=(NMIL+NALBP)*(NGR2-NGR1+1)
      NFUNC=(NMIL+NALBP+1)*(NGR2-NGR1+1)
      ALLOCATE(VARV(NPERT))
      ALLOCATE(FF(NFUNC),DFF(NPERT,NFUNC))
      CALL LCMGET(IPGRAD,'VAR-VALUE',VARV)
*----
*  COMPUTE THE RMS FUNCTIONAL AND CONSTRAINTS
*----
      IREL(:NGR2-NGR1+1)=0
      RHS(:NGR2-NGR1+1)=0.0D0
      FF(:NFUNC)=0.0D0
      WEI=REAL(NMIL)
      RMSD=0.0D0
      IPERT=0
      IFUNC=0
      DO IG=NGR1,NGR2
        SUM1=0.0D0
        DSUM=0.0D0
        DO IBM=1,NMIL
          IPERT=IPERT+1
          IFUNC=IFUNC+1
          ABS2M=MAX(EPS*AB2TOT,DBLE(ABS2(IBM,IG)))
          DELTA=ABS1(IBM,IG)*AB2TOT/(ABS2M*AB1TOT)-ABS2(IBM,IG)/ABS2M
          FF(IFUNC)=DELTA
          RMSD=RMSD+DELTA**2
          SUM1=SUM1+PHI2(IBM,IG)/VARV(IPERT)
          DSUM=DSUM+PHI2(IBM,IG)
        ENDDO
        IF(NALBP.GT.0) THEN
          OUT2M=MAX(EPSL*FI2TOT,DBLE(OUTG2(IG)))
          DELTA=OUTG1(IG)*FI2TOT/(OUT2M*FI1TOT)-OUTG2(IG)/OUT2M
          IFUNC=IFUNC+1
          FF(IFUNC)=SQRT(WEI)*DELTA
          RMSD=RMSD+WEI*DELTA**2
        ENDIF
        DELTA=SUM1/DSUM-1.0D0
        IFUNC=IFUNC+1
        FF(IFUNC)=DELTA
        RMSD=RMSD+DELTA**2
        CONST(IG-NGR1+1)=DELTA
        IPERT=IPERT+NALBP
      ENDDO
      IF(IPRINT.GT.0) THEN
        WRITE(6,100) RMSD,DOT_PRODUCT(FF,FF)
        DO IG=NGR1,NGR2
          WRITE(6,110) IG,CONST(IG-NGR1+1)
        ENDDO
      ENDIF
      IF(IPRINT.GT.2) THEN
        DO IG=1,NG
          WRITE(6,'(7H GROUP=,I4)') IG
          DO IBM=1,NMIL
            WRITE(6,120) IBM,ABS1(IBM,IG)/REAL(AB1TOT),
     1                   ABS2(IBM,IG)/REAL(AB2TOT)
          ENDDO
        ENDDO
      ENDIF
*----
*  STORE INFORMATION ON L_OPTIMIZE OBJECT
*----
      CALL LCMPUT(IPGRAD,'FOBJ-CST-VAL',NFUNC,4,FF)
      IF(LNO) GO TO 10
*----
*  COMPUTE THE GRADIENT MATRIX OF THE RMS FUNCTIONAL
*----
      ALLOCATE(SUNK(NUN))
      JPGPT=LCMLID(IPGPT,'ASOUR',NFUNC)
      IFUNC=0
      DO IG=NGR1,NGR2
        DO IBM=1,NMIL
          ABS2M=MAX(EPS*AB2TOT,DBLE(ABS2(IBM,IG)))
          SOUTOT=AB2TOT/AB1TOT/ABS2M
          SOUT2=ABS1(IBM,IG)/AB1TOT
          IFUNC=IFUNC+1
          KPGPT=LCMLIL(JPGPT,IFUNC,NG)
          DO JG=1,NG
            SUNK(:NUN)=0.0
            DO IR=1,NREG
              IUNK=KEYFLX(IR)
              IF(IUNK.EQ.0) CYCLE
              JBM=MATCOD(IR)
              IF(JBM.EQ.0) CYCLE
              SA=SIGA(JBM,JG)
              SOUT1=0.0D0
              IF((IG.EQ.JG).AND.(IBM.EQ.JBM)) SOUT1=1.0D0
              SUNK(IUNK)=REAL(SOUTOT*VOL(IR)*SA*(SOUT1-SOUT2))
            ENDDO
            CALL LCMPDL(KPGPT,JG,NUN,2,SUNK)
          ENDDO
        ENDDO
        IF(NALBP.GT.0) THEN
          IFUNC=IFUNC+1
          OUT2M=MAX(EPSL*FI2TOT,DBLE(OUTG2(IG)))
          SOUTOT=SQRT(WEI)*FI2TOT/FI1TOT/OUT2M
          SOUT2=OUTG1(IG)/FI1TOT
          KPGPT=LCMLIL(JPGPT,IFUNC,NG)
          DO JG=1,NG
            SUNK(:NUN)=0.0
            DO IR=1,NREG
              IUNK=KEYFLX(IR)
              IF(IUNK.EQ.0) CYCLE
              JBM=MATCOD(IR)
              IF(JBM.EQ.0) CYCLE
              SA=SIGA(JBM,JG)
              SF=SIGF(JBM,JG)
              SOUT1=0.0D0
              IF(IG.EQ.JG) SOUT1=1.0D0
              SUNK(IUNK)=REAL(SOUTOT*VOL(IR)*(SA*SOUT1-SF*SOUT2))
            ENDDO
            IF(IG.EQ.JG) THEN
              DO IUNK=1,NUN
                SOUT1=GAMMA(IUNK,IG)
                SUNK(IUNK)=SUNK(IUNK)+REAL(SOUTOT*SOUT1)
              ENDDO
            ENDIF
            CALL LCMPDL(KPGPT,JG,NUN,2,SUNK)
          ENDDO
        ENDIF
        IFUNC=IFUNC+1
        KPGPT=LCMLIL(JPGPT,IFUNC,NG)
        SUNK(:NUN)=0.0
        DO JG=1,NG
          CALL LCMPDL(KPGPT,JG,NUN,2,SUNK)
        ENDDO
      ENDDO
*----
*  CHECK SOURCE ORTHOGONALITY
*----
      ALLOCATE(FLUX(NUN))
      JPFLX=LCMGID(IPFLX,'FLUX')
      DO IFUNC=1,NFUNC
        KPGPT=LCMGIL(JPGPT,IFUNC)
        AIL=0.0D0
        BIL=0.0D0
        DO IG=1,NG
          CALL LCMGDL(KPGPT,IG,SUNK)
          CALL LCMGDL(JPFLX,IG,FLUX)
          DO IUNK=1,NUN
            GAZ=FLUX(IUNK)*SUNK(IUNK)
            DAZ=FLUX(IUNK)**2
            AIL=AIL+GAZ
            BIL=BIL+DAZ
          ENDDO
        ENDDO
        DSUM=ABS(AIL)/ABS(BIL)/REAL(NUN)
        IF(IPRINT.GT.3) THEN
          WRITE(6,'(/21H DRENOU: DOT PRODUCT=,1P,E11.4,11H COMPONENT=,
     1    I5)') DSUM,IFUNC
        ENDIF
        IF(ABS(DSUM).GT.1.0E-4) THEN
          WRITE(HSMG,'(36HDRENOU: NON ORTHOGONAL SOURCE (DSUM=,1P,E11.3,
     1    26H) FOR INDIVIDUAL COMPONENT,I5,1H.)') DSUM,IFUNC
          CALL XABORT(HSMG)
        ENDIF
      ENDDO
      DEALLOCATE(FLUX,SUNK)
*----
*  COMPUTE THE DIRECT GRADIENT MATRIX
*----
      IFUNC=0
      DFF(:NPERT,:NFUNC)=0.0D0
      DO IG=NGR1,NGR2
        DSUM=0.0D0
        DO IBM=1,NMIL
          DSUM=DSUM+PHI2(IBM,IG)
        ENDDO
        DO IBM=1,NMIL
          ABS2M=MAX(EPS*AB2TOT,DBLE(ABS2(IBM,IG)))
          SOUTOT=ABS1(IBM,IG)*AB2TOT/AB1TOT/ABS2M
          IFUNC=IFUNC+1
          IPERT=0
          DO JG=NGR1,NGR2
            DO JBM=1,NMIL
              IPERT=IPERT+1
              IF((IG.EQ.JG).AND.(IBM.EQ.JBM)) THEN
                DFF(IPERT,IFUNC)=DFF(IPERT,IFUNC)+SOUTOT/VARV(IPERT)
              ENDIF
              DFF(IPERT,IFUNC)=DFF(IPERT,IFUNC)-SOUTOT*ABS1(JBM,JG)/
     >        AB1TOT/VARV(IPERT)
            ENDDO
            IF(NALBP.GT.0) IPERT=IPERT+1
          ENDDO
        ENDDO
        IF(NALBP.GT.0) THEN
          IFUNC=IFUNC+1
          IPERT=0
          OUT2M=MAX(EPSL*FI2TOT,DBLE(OUTG2(IG)))
          SOUTOT=SQRT(WEI)*FI2TOT/FI1TOT/OUT2M
          DO JG=NGR1,NGR2
            DO JBM=1,NMIL
              IPERT=IPERT+1
              IF(IG.EQ.JG) THEN
                DFF(IPERT,IFUNC)=DFF(IPERT,IFUNC)+SOUTOT*ABS1(JBM,JG)/
     >          VARV(IPERT)
              ENDIF
              DFF(IPERT,IFUNC)=DFF(IPERT,IFUNC)-SOUTOT*OUTG1(IG)*
     >        SUM(LHS1(JBM,:NG,JG))/FI1TOT/VARV(IPERT)
            ENDDO
            IPERT=IPERT+1
          ENDDO
        ENDIF
        IFUNC=IFUNC+1
        IPERT=0
        DO JG=NGR1,NGR2
          DO JBM=1,NMIL
            IPERT=IPERT+1
            IF(IG.EQ.JG) THEN
              DFF(IPERT,IFUNC)=DFF(IPERT,IFUNC)-PHI2(JBM,IG)/(DSUM*
     >        VARV(IPERT)**2)
            ENDIF
          ENDDO
          IF(NALBP.GT.0) IPERT=IPERT+1
        ENDDO
      ENDDO
      CALL LCMPUT(IPGRAD,'GRADIENT-DIR',NPERT*NFUNC,4,DFF)
*----
*  MODIFY STATE VECTOR OF OPTIMIZE OBJECT
*----
   10 CALL LCMGET(IPGRAD,'STATE-VECTOR',ISTATE)
      ISTATE(2)=NFUNC-1
      ISTATE(8)=4
      CALL LCMPUT(IPGRAD,'STATE-VECTOR',NSTATE,1,ISTATE)
*----
*  SCRATCH STORAGE DEALLOCATION
*----
      DEALLOCATE(DFF,FF)
      DEALLOCATE(VARV,SIGF,SIGA,OUTG2R,OUTG2,OUTG1,GAMMA,RHS,IREL,CONST,
     1 LHS2,RHS2,LHS1,RHS1,ABS2,ABS1,PHI2,PHI1)
      RETURN
*
  100 FORMAT(/40H DRENOU: RMS ERROR ON RATE DISTRIBUTION=,1P,2E11.4)
  110 FORMAT(23H DRENOU:    CONSTRAINT(,I4,2H)=,1P,E11.4)
  120 FORMAT(5X,16HABSORPTION RATE(,I4,2H)=,1P,2E12.4)
  130 FORMAT(5X,6HGROUP=,I4,9H LEAKAGE=,1P,2E12.4)
      END
