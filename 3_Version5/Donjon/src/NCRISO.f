*DECK NCRISO
      SUBROUTINE NCRISO(IPLIB,LPCPO,NBISO1,IMICR,HNAME,JSO,IBM,NCAL,
     1 NGRP,NL,NW,NED,HVECT,NDEL,NDFI,IMPX,FACT,TERP,LPURE)
*
*-----------------------------------------------------------------------
*
*Purpose:
* Recover nuclear data from a single isotopic directory.
*
*Copyright:
* Copyright (C) 2006 Ecole Polytechnique de Montreal
*
*Author(s): 
* A. Hebert
*
*Parameters: input
* IPLIB   address of the microlib LCM object.
* LPCPO   address of the 'CALCULATIONS' tree in the multidimensional
*         multicompo object.
* NBISO1  number of multicompo isotopes.
* IMICR   index of microlib isotope corresponding to each multicompo
*         isotope in mixture IBM.
* HNAME   character*12 name of the multicompo isotope been processed.
* JSO     index of the multicompo isotope been processed.
* IBM     mixture index.
* NCAL    number of elementary calculations in the multicompo object.
* NGRP    number of energy groups.
* NL      number of Legendre orders.
* NW      type of weighting for P1 cross section info (=0 P0; =1 P1).
* NED     number of extra vector edits.
* HVECT   character names of the extra vector edits.
* NDEL    number of delayed precursor groups.
* NDFI    number of fissile isotopes.
* IMPX    print parameter (equal to zero for no print).
* FACT    number density factors.
* TERP    interpolation weights.
* LPURE   flag set to .true. to avoid non-linear interpolation effects.
*
*-----------------------------------------------------------------------
*
      USE GANLIB
      IMPLICIT NONE
*----
*  SUBROUTINE ARGUMENTS
*----
      TYPE(C_PTR) IPLIB,LPCPO
      INTEGER NBISO1,IMICR(NBISO1),JSO,IBM,NCAL,NGRP,NL,NW,NED,NDEL,
     1 NDFI,IMPX
      REAL FACT(NCAL),TERP(NCAL)
      CHARACTER HNAME*12,HVECT(NED)*(*)
      LOGICAL LPURE
*----
*  LOCAL VARIABLES
*----
      INTEGER, PARAMETER::IOUT=6
      REAL AWR, DECAY, EMEVF, EMEVG, FACT0, TAUXFI, TAUXF, WEIGHT
      INTEGER ICAL, IDEL, IED, IFI, IG1, IG2, IG, ILENG, IL, ITYLCM, J,
     & LENGTH, IW, MAXH, IOF
      LOGICAL LAWR,LMEVF,LMEVG,LDECA,LWD,LYIELD,LPIFI
      CHARACTER CM*2,TEXT12*12
      TYPE(C_PTR) MPCPO,NPCPO,OPCPO
      INTEGER, ALLOCATABLE, DIMENSION(:) :: JPIF1,JPIF2,ITYPR
      REAL, ALLOCATABLE, DIMENSION(:) :: YIEL1,PYIE1,YIEL2,PYIE2,WDLA
      REAL, ALLOCATABLE, DIMENSION(:,:) :: GAR1,GAR2
      REAL, ALLOCATABLE, DIMENSION(:,:,:) :: WSCA1,WSCA2
      CHARACTER(LEN=12), ALLOCATABLE, DIMENSION(:) :: HMAKE
*----
*  SCRATCH STORAGE ALLOCATION
*----
      MAXH=9+3*NW+NL+NED+2*NDEL
      ALLOCATE(JPIF1(NDFI),JPIF2(NDFI),ITYPR(NL))
      ALLOCATE(GAR1(NGRP,MAXH),YIEL1(NGRP+1),PYIE1(NDFI),
     1 WSCA1(NGRP,NGRP,NL),GAR2(NGRP,MAXH),YIEL2(NGRP+1),PYIE2(NDFI),
     2 WSCA2(NGRP,NGRP,NL),WDLA(NDEL))
      ALLOCATE(HMAKE(MAXH+NL))
*----
*  RECOVER GENERIC ISOTOPIC DATA FROM THE MULTICOMPO
*----
      LAWR=.FALSE.
      LMEVF=.FALSE.
      LMEVG=.FALSE.
      LDECA=.FALSE.
      LYIELD=.FALSE.
      LPIFI=.FALSE.
      LWD=.FALSE.
      DO 10 ICAL=1,NCAL
      MPCPO=LCMGIL(LPCPO,ICAL)
      CALL LCMLEN(MPCPO,'ISOTOPESLIST',LENGTH,ITYLCM)
      IF(LENGTH.EQ.0) GO TO 10
      NPCPO=LCMGID(MPCPO,'ISOTOPESLIST')
      CALL LCMLEL(NPCPO,JSO,ILENG,ITYLCM)
      IF(ILENG.EQ.0) GO TO 10
      OPCPO=LCMGIL(NPCPO,JSO)
      CALL LCMGTC(OPCPO,'ALIAS',12,TEXT12)
      IF(TEXT12(:8).NE.HNAME(:8)) GO TO 10
      CALL LCMLEN(OPCPO,'AWR',LENGTH,ITYLCM)
      IF(LENGTH.EQ.1) CALL LCMGET(OPCPO,'AWR',AWR)
      LAWR=(LENGTH.EQ.1)
      CALL LCMLEN(OPCPO,'MEVF',LENGTH,ITYLCM)
      IF(LENGTH.EQ.1) CALL LCMGET(OPCPO,'MEVF',EMEVF)
      LMEVF=(LENGTH.EQ.1)
      CALL LCMLEN(OPCPO,'MEVG',LENGTH,ITYLCM)
      IF(LENGTH.EQ.1) CALL LCMGET(OPCPO,'MEVG',EMEVG)
      LMEVG=(LENGTH.EQ.1)
      CALL LCMLEN(OPCPO,'DECAY',LENGTH,ITYLCM)
      IF(LENGTH.EQ.1) CALL LCMGET(OPCPO,'DECAY',DECAY)
      LDECA=(LENGTH.EQ.1)
      CALL LCMLEN(OPCPO,'LAMBDA-D',LENGTH,ITYLCM)
      LWD=(LENGTH.EQ.NDEL).AND.(NDEL.GT.0)
      IF(LWD) CALL LCMGET(OPCPO,'LAMBDA-D',WDLA)
      GO TO 15
   10 CONTINUE
      WRITE(6,170) IBM,HNAME
      CALL XABORT('NCRISO: UNABLE TO FIND AN ISOTOPE DIRECTORY.')
*----
*  LOOP OVER ELEMENTARY CALCULATIONS
*----
   15 DO J=1,MAXH+NL
         HMAKE(J)=' '
      ENDDO
      GAR2(:NGRP,:MAXH)=0.0
      WSCA2(:NGRP,:NGRP,:NL)=0.0
      YIEL2(:NGRP+1)=0.0
      PYIE2(:NDFI)=0.0
      JPIF2(:NDFI)=0
      TAUXFI=0.0
      DO 120 ICAL=1,NCAL
      WEIGHT=TERP(ICAL)
      IF(WEIGHT.EQ.0.0) GO TO 120
      FACT0=FACT(ICAL)
      MPCPO=LCMGIL(LPCPO,ICAL)
      IF(IMPX.GT.4) THEN
         WRITE(IOUT,'(39H NCRISO: MULTICOMPO ACCESS FOR ISOTOPE ,A,
     1   16H AND CALCULATION,I5,1H.)') HNAME,ICAL
         IF(IMPX.GT.50) CALL LCMLIB(MPCPO)
      ENDIF
      NPCPO=LCMGID(MPCPO,'ISOTOPESLIST')
      CALL LCMLEL(NPCPO,JSO,ILENG,ITYLCM)
      IF(ILENG.EQ.0) GO TO 120
      OPCPO=LCMGIL(NPCPO,JSO)
*----
*  RECOVER CALCULATION-SPECIFIC ISOTOPIC DATA FROM THE MULTICOMPO
*----
      DO IW=1,MIN(NW+1,10)
         WRITE(TEXT12,'(3HNWT,I1)') IW-1
         CALL LCMLEN(OPCPO,TEXT12,LENGTH,ITYLCM)
         IF(LENGTH.EQ.NGRP) THEN
            CALL LCMGET(OPCPO,TEXT12,GAR1(1,IW))
            HMAKE(IW)=TEXT12
         ENDIF
         WRITE(TEXT12,'(4HNWAT,I1)') IW-1
         CALL LCMLEN(OPCPO,TEXT12,LENGTH,ITYLCM)
         IF(LENGTH.EQ.NGRP)  THEN
            CALL LCMGET(OPCPO,TEXT12,GAR1(1,1+NW+IW))
            HMAKE(1+NW+IW)=TEXT12
         ENDIF
         WRITE(TEXT12,'(4HNTOT,I1)') IW-1
         CALL LCMLEN(OPCPO,TEXT12,LENGTH,ITYLCM)
         IF(LENGTH.EQ.NGRP)  THEN
            CALL LCMGET(OPCPO,TEXT12,GAR1(1,2+2*NW+IW))
            HMAKE(2+2*NW+IW)=TEXT12
         ENDIF
      ENDDO
      CALL XDRLGS(OPCPO,-1,IMPX,0,NL-1,1,NGRP,GAR1(1,4+3*NW),WSCA1,
     1 ITYPR)
      DO IL=0,NL-1
         IF(ITYPR(IL+1).NE.0) THEN
            WRITE (CM,'(I2.2)') IL
            HMAKE(4+3*NW+IL)='SIGS'//CM
         ENDIF
      ENDDO
      CALL LCMLEN(OPCPO,'NUSIGF',LENGTH,ITYLCM)
      IF(LENGTH.EQ.NGRP) THEN
         CALL LCMGET(OPCPO,'NUSIGF',GAR1(1,4+3*NW+NL))
         HMAKE(4+3*NW+NL)='NUSIGF'
         CALL LCMGET(OPCPO,'CHI',GAR1(1,5+3*NW+NL))
         HMAKE(5+3*NW+NL)='CHI'
      ENDIF
      IF(NDEL.GT.0) THEN
         WRITE(TEXT12,'(6HNUSIGF,I2.2)') NDEL
         CALL LCMLEN(OPCPO,TEXT12,LENGTH,ITYLCM)
         IF(LENGTH.EQ.NGRP) THEN
            DO IDEL=1,NDEL
               WRITE(TEXT12,'(6HNUSIGF,I2.2)') IDEL
               CALL LCMGET(OPCPO,TEXT12,GAR1(1,5+3*NW+NL+IDEL))
               HMAKE(5+3*NW+NL+IDEL)=TEXT12
            ENDDO
         ENDIF
         WRITE(TEXT12,'(3HCHI,I2.2)') NDEL
         CALL LCMLEN(OPCPO,TEXT12,LENGTH,ITYLCM)
         IF(LENGTH.EQ.NGRP) THEN
            DO IDEL=1,NDEL
               WRITE(TEXT12,'(3HCHI,I2.2)') IDEL
               CALL LCMGET(OPCPO,TEXT12,GAR1(1,5+3*NW+NL+NDEL+IDEL))
               HMAKE(5+3*NW+NL+NDEL+IDEL)=TEXT12
            ENDDO
         ENDIF
      ENDIF
      CALL LCMLEN(OPCPO,'H-FACTOR',LENGTH,ITYLCM)
      IF(LENGTH.EQ.NGRP) THEN
         CALL LCMGET(OPCPO,'H-FACTOR',GAR1(1,6+3*NW+NL+2*NDEL))
         HMAKE(6+3*NW+NL+2*NDEL)='H-FACTOR'
      ENDIF
      CALL LCMLEN(OPCPO,'OVERV',LENGTH,ITYLCM)
      IF(LENGTH.EQ.NGRP) THEN
         CALL LCMGET(OPCPO,'OVERV',GAR1(1,7+3*NW+NL+2*NDEL))
         HMAKE(7+3*NW+NL+2*NDEL)='OVERV'
      ENDIF
      CALL LCMLEN(OPCPO,'TRANC',LENGTH,ITYLCM)
      IF(LENGTH.EQ.NGRP) THEN
         CALL LCMGET(OPCPO,'TRANC',GAR1(1,8+3*NW+NL+2*NDEL))
         HMAKE(8+3*NW+NL+2*NDEL)='TRANC'
      ENDIF
      DO IED=1,NED
         CALL LCMLEN(OPCPO,HVECT(IED),LENGTH,ITYLCM)
         IF((LENGTH.GT.0).AND.(HVECT(IED).NE.'TRANC')) THEN
            CALL LCMGET(OPCPO,HVECT(IED),GAR1(1,8+3*NW+NL+2*NDEL+IED))
            HMAKE(8+3*NW+NL+2*NDEL+IED)=HVECT(IED)
         ENDIF
      ENDDO
      CALL LCMLEN(OPCPO,'STRD',LENGTH,ITYLCM)
      IF(LENGTH.EQ.NGRP) THEN
         CALL LCMGET(OPCPO,'STRD',GAR1(1,MAXH))
         HMAKE(MAXH)='STRD'
      ENDIF
*----
*  RECOVER FISSION YIELD DATA
*----
      CALL LCMLEN(OPCPO,'YIELD',LENGTH,ITYLCM)
      IF(LENGTH.EQ.NGRP+1) THEN
         CALL LCMGET(OPCPO,'YIELD',YIEL1)
         LYIELD=.TRUE.
         DO IG=1,NGRP+1
           YIEL2(IG)=YIEL2(IG)+WEIGHT*YIEL1(IG)
         ENDDO
      ENDIF
      CALL LCMLEN(OPCPO,'PYIELD',LENGTH,ITYLCM)
      IF((LENGTH.GT.0).AND.(LENGTH.EQ.NDFI)) THEN
         CALL LCMGET(OPCPO,'PIFI',JPIF1)
         CALL LCMGET(OPCPO,'PYIELD',PYIE1)
         LPIFI=.TRUE.
         DO IFI=1,NDFI
            IF(JPIF1(IFI).GT.0) JPIF2(IFI)=IMICR(JPIF1(IFI))
            PYIE2(IFI)=PYIE2(IFI)+WEIGHT*PYIE1(IFI)
         ENDDO
      ENDIF
*----
*  COMPUTE FISSION RATE FOR A SINGLE ELEMENTARY CALCULATION
*----
      TAUXF=0.0
      IF(HMAKE(4+3*NW+NL).EQ.'NUSIGF') THEN
        DO IG=1,NGRP
           TAUXF=TAUXF+GAR1(IG,4+3*NW+NL)*GAR1(IG,1)
        ENDDO
        TAUXFI=TAUXFI+FACT0*WEIGHT*TAUXF
      ENDIF
*----
*  ADD CONTRIBUTIONS FROM A SINGLE ELEMENTARY CALCULATION
*----
      DO J=1,MAXH
         IF((HMAKE(J).NE.' ').AND.(HMAKE(J)(:4).NE.'SIGS')) THEN
            DO IG=1,NGRP
               IF((HMAKE(J)(:2).EQ.'NW').OR.(HMAKE(J).EQ.'OVERV')) THEN
                  GAR2(IG,J)=GAR2(IG,J)+WEIGHT*GAR1(IG,J)
               ELSE IF((HMAKE(J)(:3).EQ.'CHI').AND.(.NOT.LPURE)) THEN
                  GAR2(IG,J)=GAR2(IG,J)+FACT0*WEIGHT*TAUXF*GAR1(IG,J)
               ELSE
                  GAR2(IG,J)=GAR2(IG,J)+FACT0*WEIGHT*GAR1(IG,J)
               ENDIF
            ENDDO
         ENDIF
      ENDDO
      DO IL=1,NL
         IOF=3+3*NW+IL
         ITYPR(IL)=0
         IF(HMAKE(MAXH+IL).NE.' ') ITYPR(IL)=1
         DO IG2=1,NGRP
            GAR2(IG2,IOF)=GAR2(IG2,IOF)+FACT0*WEIGHT*GAR1(IG2,IOF)
            DO IG1=1,NGRP
               WSCA2(IG1,IG2,IL)=WSCA2(IG1,IG2,IL)+FACT0*WEIGHT*
     1         WSCA1(IG1,IG2,IL)
            ENDDO
         ENDDO
      ENDDO
  120 CONTINUE
*----
*  NORMALIZE FISSION SPECTRA
*----
      IF(.NOT.LPURE) THEN
        DO J=1,MAXH
           IF(HMAKE(J)(:3).EQ.'CHI') THEN
              DO IG=1,NGRP
                 IF(GAR2(IG,J).NE.0.0) GAR2(IG,J)=GAR2(IG,J)/TAUXFI
              ENDDO
           ENDIF
        ENDDO
      ENDIF
*----
*  SAVE ISOTOPIC DATA IN THE MICROLIB
*----
      CALL LCMPTC(IPLIB,'ALIAS',12,HNAME)
      IF(LAWR) CALL LCMPUT(IPLIB,'AWR',1,2,AWR)
      IF(LMEVF) CALL LCMPUT(IPLIB,'MEVF',1,2,EMEVF)
      IF(LMEVG) CALL LCMPUT(IPLIB,'MEVG',1,2,EMEVG)
      IF(LDECA) CALL LCMPUT(IPLIB,'DECAY',1,2,DECAY)
      IF(LYIELD) CALL LCMPUT(IPLIB,'YIELD',NGRP+1,2,YIEL2)
      IF(LPIFI) THEN
         CALL LCMPUT(IPLIB,'PYIELD',NDFI,2,PYIE2)
         CALL LCMPUT(IPLIB,'PIFI',NDFI,1,JPIF2)
      ENDIF
      IF(LWD) CALL LCMPUT(IPLIB,'LAMBDA-D',NDEL,2,WDLA)
      DO J=1,MAXH
         IF((HMAKE(J).NE.' ').AND.(HMAKE(J)(:4).NE.'SIGS')) THEN
            CALL LCMPUT(IPLIB,HMAKE(J),NGRP,2,GAR2(1,J))
         ENDIF
      ENDDO
      CALL XDRLGS(IPLIB,1,IMPX,0,NL-1,1,NGRP,GAR2(1,4+3*NW),WSCA2,ITYPR)
      IF(IMPX.GT.50) CALL LCMLIB(IPLIB)
*----
*  SCRATCH STORAGE DEALLOCATION
*----
      DEALLOCATE(HMAKE)
      DEALLOCATE(WDLA,WSCA2,PYIE2,YIEL2,GAR2,WSCA1,PYIE1,YIEL1,GAR1)
      DEALLOCATE(ITYPR,JPIF2,JPIF1)
      RETURN
*
  170 FORMAT(17H NCRISO: MIXTURE=,I5,10H ISOTOPE=',A12,2H'.)
      END
