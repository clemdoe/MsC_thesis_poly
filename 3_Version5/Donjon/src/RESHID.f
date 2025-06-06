*DECK RESHID
      SUBROUTINE RESHID(IPMAP,IPMTX,NX,NZ,LX,LZ,MIX,NFUEL,IMPX)
*
*-----------------------------------------------------------------------
*
*Purpose:
* Update material index, it will store the negative fuel mixtures.
*
*Copyright:
* Copyright (C) 2007 Ecole Polytechnique de Montreal.
*
*Author(s): 
* V. Descotes
*
*Parameters: input/output
* IPMAP  pointer to fuel-map information.
* IPMTX  pointer to matex information.
* NX     number of elements along x-axis in fuel map.
* NZ     number of elements along z-axis in fuel map.
* LX     number of elements along x-axis in geometry.
* LZ     number of elements along z-axis in geometry.
* MIX    renumbered index over the fuel-map geometry.
* NFUEL  number of fuel types.
* IMPX   printing index (=0 for no print).
*
*-----------------------------------------------------------------------
*
      USE GANLIB
*----
*  SUBROUTINE ARGUMENTS
*----
      TYPE(C_PTR) IPMAP,IPMTX
      INTEGER NX,NZ,LX,LZ,MIX(NX*NZ),NFUEL,IMPX
*----
*  LOCAL VARIABLES
*----
      PARAMETER(IOUT=6)
      INTEGER ISPLTY(1),NCODE(6)
      REAL    MTXSIDE,MAPSIDE
      TYPE(C_PTR) JPMAP
*----
*  ALLOCATABLE ARRAYS
*----
      INTEGER, ALLOCATABLE, DIMENSION(:) :: IMAT,ISPLTX,ISPLTZ,INDX,
     1 FTOT,DPP,MX
      REAL, ALLOCATABLE, DIMENSION(:) :: MAPZZ,GEOZZ
*----
*  SCRATCH STORAGE ALLOCATION
*----
      ALLOCATE(ISPLTX(LX),ISPLTZ(LZ),INDX(LX*LZ),FTOT(NFUEL))
      ALLOCATE(MAPZZ(NZ+1),GEOZZ(LZ+1))
*----
*  RECOVER GEOMETRY AND FUELMAP INFORMATION
*----
      CALL LCMGET(IPMTX,'SIDE',MTXSIDE)
      CALL LCMGET(IPMTX,'MESHZ',GEOZZ)
      JPMAP=LCMGID(IPMAP,'GEOMAP')
      CALL LCMGET(JPMAP,'IHEX',IHEX)
      CALL LCMGET(JPMAP,'SIDE',MAPSIDE)
      CALL LCMGET(JPMAP,'MESHZ',MAPZZ)
      ISPLTL=0
      CALL LCMLEN(JPMAP,'SPLITL',ILONG,ITYLCM)
      IF(ILONG.NE.0) CALL LCMGET(JPMAP,'SPLITL',ISPLTL)
*----
*  UNFOLD GEOMETRY IF HEXAGONAL IN LOZENGES
*----
      IF((ISPLTL.GT.0).AND.(IHEX.NE.9)) THEN
         MAXPTS=LX*LZ
         ALLOCATE(DPP(MAXPTS),MX(NX*NZ))
         DO 10 I=1,NX*NZ
         MX(I)=MIX(I)
   10    CONTINUE
         NXOLD=NX
         CALL BIVALL(MAXPTS,IHEX,NXOLD,NX,DPP)
         DO 30 KZ=1,NZ
         DO 20 KX=1,NX
         KEL=DPP(KX)+(KZ-1)*NXOLD
         INDX(KX+(KZ-1)*NX)=MX(KEL)
   20    CONTINUE
   30    CONTINUE
         DEALLOCATE(DPP,MX)
         IHEX=9
      ELSE
         INDX(:NX*NZ)=MIX(:NX*NZ)
      ENDIF
*----
*  FUELMAP INFORMATION SPLITTING
*----
      NY=1
      ITYPE=9
      ISPLTX(:NX)=1
      ISPLTY(:NY)=1
      IZ=1
      DO KM=1,NZ
        ISPLTZ(KM)=0
        DO JZ=IZ,LZ
          IF(GEOZZ(JZ+1).LE.MAPZZ(KM+1)) THEN
            ISPLTZ(KM)=ISPLTZ(KM)+1
          ELSE
            IZ=JZ
            EXIT
          ENDIF
        ENDDO
      ENDDO
      MAXPTS=LX*LZ
      LX1=LX
      LY1=1
      LZ1=LZ
      CALL SPLIT0 (MAXPTS,ITYPE,NCODE,NX,NY,NZ,ISPLTX,ISPLTY,ISPLTZ,
     1 0,ISPLTL,NMBLK,LX1,LY1,LZ1,MAPSIDE,XXX,YYY,ZZZ,INDX,.FALSE.,
     2 IMPX)
      IF(ISPLTL.GT.0) MAPSIDE=MAPSIDE/REAL(ISPLTL)
      IF(ABS(MAPSIDE-MTXSIDE).GT.1.0E-6) CALL XABORT('RESHID: INVALID '
     1 //'SIDE.')
*     CHECK TOTAL NUMBER
      ITOT=0
      DO 40 IEL=1,LX*LZ
      IF(INDX(IEL).NE.0)ITOT=ITOT+1
   40 CONTINUE
      NTOT=0
      CALL LCMGET(IPMTX,'FTOT',FTOT)
      DO 50 IFUEL=1,NFUEL
      NTOT=NTOT+FTOT(IFUEL)
   50 CONTINUE
      IF(ITOT.NE.NTOT) THEN
         WRITE(IOUT,'(/15H @RESHID: ITOT=,I8,6H NTOT=,I8)') ITOT,NTOT
         CALL XABORT('@RESHID: FOUND DIFFERENT TOTAL NUMBER OF FUEL MI'
     1   //'XTURES IN FUEL-MAP AND MATEX.')
      ENDIF
*     STORE NEGATIVE FUEL MIXTURES
      CALL LCMLEN(IPMTX,'MAT',LENGT,ITYP)
      ALLOCATE(IMAT(LENGT))
      IMAT(:LENGT)=0
      CALL LCMGET(IPMTX,'MAT',IMAT)
      DO 60 IEL=1,LX*LZ
      IF(INDX(IEL).NE.0)IMAT(IEL)=-INDX(IEL)
   60 CONTINUE
      CALL LCMPUT(IPMTX,'MAT',LENGT,1,IMAT)
*----
*  SCRATCH STORAGE DEALLOCATION
*----
      DEALLOCATE(IMAT,GEOZZ,MAPZZ,FTOT,INDX,ISPLTZ,ISPLTX)
      RETURN
      END
