import { Component } from '@angular/core';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  title = 'dogs';

  showImageHappy: boolean;
  showImageSad: boolean;
  showImageAngry: boolean;
  showImageRelax: boolean;
  url: any;

  constructor() {
    this.showImageHappy = false;
    this.showImageSad = false;
    this.showImageAngry = false;
    this.showImageRelax = false;
  }
  
  uploadFile($event:any) {
    this.uploadDocument($event.target.files[0]);
  }

  uploadDocument(file:any) {
    let fileReader = new FileReader();
    fileReader.onload = (e) => {
      this.url = fileReader.result; 
    }
    return fileReader.readAsDataURL(file);
}
}
