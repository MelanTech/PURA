filename1="/data/zzf/proj/TBSI/output/test/tracking_results/tbsi_track/vitb_256_tbsi_32x1_1e4_lasher_15ep_sot/lasher/"
if [ -d "$filename1" ]; then
  num=$(ls -l "$filename1" | grep "^-" | wc -l)
  printf "%-10s %4d/245 \n" lasher_test $num
fi

filename2="/data/zzf/proj/TBSI/output/test/tracking_results/tbsi_track/vitb_256_tbsi_32x1_1e4_lasher_15ep_sot/vtuav"
if [ -d "$filename2" ]; then
  num=$(ls -l "$filename1" | grep "^-" | wc -l)
  printf "%-10s %4d/250 \n" vtuav_test $num
fi

filename3="/data/zzf/proj/TBSI/output/test/tracking_results/tbsi_track/vitb_256_tbsi_32x1_1e4_lasher_15ep_sot/rgbt234"
if [ -d "$filename3" ]; then
  num=$(ls -l "$filename3" | grep "^-" | wc -l)
  printf "%-10s %4ds/234" rgbt234 $num
fi

filename4="/data/zzf/proj/TBSI/output/test/tracking_results/tbsi_track/vitb_256_tbsi_32x1_1e4_lasher_15ep_sot_cls/lasher/"
if [ -d "$filename4" ]; then
  num=$(ls -l "$filename4" | grep "^-" | wc -l)
  printf "cls: %-10s %4d/245 \n" lasher_test $num
fi

filename5="/data/zzf/proj/TBSI/output/test/tracking_results/tbsi_track/vitb_256_tbsi_32x1_1e4_lasher_15ep_sot_cls/vtuav"
if [ -d "$filename5" ]; then
  num=$(ls -l "$filename5" | grep "^-" | wc -l)
  printf "cls: %-10s %4d/250 \n" vtuav_test $num
fi

filename6="/data/zzf/proj/TBSI/output/test/tracking_results/tbsi_track/vitb_256_tbsi_32x1_1e4_lasher_15ep_sot_cls/rgbt234"
if [ -d "$filename6" ]; then
  num=$(ls -l "$filename6" | grep "^-" | wc -l)
  printf "cls: %-10s  %4d/234\n" rgbt234 $num
fi

filename7="/data/zzf/proj/TBSI/output/test/tracking_results/tbsi_track/vitb_256_tbsi_32x1_1e4_lasher_15ep_sot_cls/rgbt210"
if [ -d "$filename7" ]; then
  num=$(ls -l "$filename7" | grep "^-" | wc -l)
  printf "cls: %-10s  %4d/210" rgbt210 $num
fi

filename8="/data/zzf/proj/TBSI/output/test/tracking_results/tbsi_track/vitb_256_tbsi_32x1_1e4_lasher_15ep_sot_cls/gtot"
if [ -d "filename8" ]; then
  num=$(ls -l "$filename8" | grep "^-" | wc -l)
  printf "cls: %-10s  %4d/210" gtot $num
fi